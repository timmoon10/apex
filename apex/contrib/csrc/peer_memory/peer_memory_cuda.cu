#include <torch/extension.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CUDAContext.h>
#include <list>
#include <cstdio>
#include <cassert>
#include <cuda_runtime_api.h>
#include "nccl.h"

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if( err != cudaSuccess ) {                        \
    char hostname[1024];                            \
    gethostname(hostname, 1024);                    \
    printf("%s: CUDA failure %s:%d '%s'\n",         \
         hostname,                                  \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
  }                                                 \
} while(0)

namespace {

constexpr int THREADS_PER_CTA = 128;

/* Basic deleter function for from_blob function.
void deleter(void* ptr)
{
    printf("deleter(ptr=%p)\n",ptr);
    cudaFree(ptr);
}
*/

template<class T>
at::Tensor blob_view(T* raw_ptr, std::vector<int64_t> shape, const at::TensorOptions& options, bool channels_last)
{
    size_t size = 1;
    std::vector<int64_t> strides(shape.size());
    if (channels_last) {
        assert(shape.size() == 4);
        strides[0] = shape[1]*shape[2]*shape[3];
        strides[1] = 1;
        strides[2] = shape[1]*shape[3];
        strides[3] = shape[1];
    } else {
        int idx = strides.size();
        for (auto it = shape.rbegin();  it != shape.rend();  ++it)
        {
	    strides[--idx] = size;
	    size *= *it;
        }
    }
    size *= sizeof(T);
    // TODO: Implement dynamic reuse of pooled peer memory.
    // We provide no deleter function because all peer memory allocations are static in this implementation.
    return torch::from_blob((void*)raw_ptr, shape, strides, 0L, options);
}

void tensor_shape(at::Tensor t, bool explicit_nhwc, int& N, int& C, int& H, int& W)
{
    if (t.dim() == 3) {
	N = 1;
        if (explicit_nhwc) {
            C = t.size(2);
            H = t.size(0);
            W = t.size(1);
        } else {
	    C = t.size(0);
    	    H = t.size(1);
    	    W = t.size(2);
        }
    } else if (t.dim() == 4) {
        if (explicit_nhwc) {
            N = t.size(0);
            C = t.size(3);
            H = t.size(1);
            W = t.size(2);
        } else {
            N = t.size(0);
            C = t.size(1);
            H = t.size(2);
            W = t.size(3);
        }
    } else {
        printf("%s;%d - t.dim() must be either 3 or 4 (was %d)\n",__FILE__,__LINE__,t.dim());
        assert(t.dim() == 3 || t.dim() == 4);
    }
}

void tensor_strides(at::Tensor t, bool explicit_nhwc, int& stride_N, int& stride_C, int& stride_H, int& stride_W)
{
    if (t.dim() == 3) {
        if (explicit_nhwc) {
            stride_C = t.stride(2);
            stride_H = t.stride(0);
            stride_W = t.stride(1);
        } else {
	    stride_C = t.stride(0);
    	    stride_H = t.stride(1);
    	    stride_W = t.stride(2);
        }
        stride_N = t.size(0)*t.size(1)*t.size(2);
    } else if (t.dim() == 4) {
        if (explicit_nhwc) {
            stride_N = t.stride(0);
            stride_C = t.stride(3);
            stride_H = t.stride(1);
            stride_W = t.stride(2);
        } else {
            stride_N = t.stride(0);
            stride_C = t.stride(1);
            stride_H = t.stride(2);
            stride_W = t.stride(3);
        }
    } else {
        printf("%s;%d - t.dim() must be either 3 or 4 (was %d)\n",__FILE__,__LINE__,t.dim());
        assert(t.dim() == 3 || t.dim() == 4);
    }
}

template<class T>
inline __device__ void __zero(T* dst)
{
    *dst = T(0);
}

inline __device__ void __zero(int2* dst)
{
    *dst = {0, 0};
}

template<class T, bool channels_last, bool zero>
inline __device__ void push_pull_tensor(
        const T* __restrict__ data_in,  // local memory
        const int data_in_stride_C,
        const int data_in_stride_H,
        const int data_in_stride_W,
        int4* transfer_out,             // remote peer memory
        int4* transfer_in,              // local peer memory
	T* __restrict__ data_out,       // local memory
        const int data_out_stride_C,
        const int data_out_stride_H,
        const int data_out_stride_W,
	const int NC,
        const int NH,
        const int NW,
        const int thread_id,
        const int num_threads
	)
{
    const int count = NC*NH*NW;

    // communicate in 128b chunks
    // Note: NVLink flit size is 128b=16B. Use last 4B as a semaphore.
    static_assert(sizeof(T) <= 12);
    union Flit {
        T payload;
        uint uints[4];
    };

    // transfer buffers are contiguous
    int transfer_stride_C, transfer_stride_H, transfer_stride_W;
    if (channels_last) {
      transfer_stride_C = 1;
      transfer_stride_H = NC;
      transfer_stride_W = NW*NC;
    } else {
      transfer_stride_C = NH*HW;
      transfer_stride_H = NW;
      transfer_stride_W = 1;
    }

    // send data to peer GPU
    if (!zero) {
        for (int i = thread_id;  i < count;  i += num_threads) {
            // calculate position in buffers
            int c, h, w;
            if (channels_last) {
                const int j = i / NC;
                c = i % NC;
                h = j / NW;
                w = j % NW;
            } else {
                const int j = i / NW;
                w = i % NW;
                c = j / NH;
                h = j % NH;
            }
            const T& in = data_in[c*data_in_stride_C + h*data_in_stride_H + w*data_in_stride_W];
            int4& out = transfer_out[c*transfer_stride_C + h*transfer_stride_H + w*transfer_stride_W];

            // pack value into flit
            Flit flit;
            flit.payload = in;
            flit.uints[3] = 0xffffffff;

            // make sure peer buffer is ready
            // TODO Use double-buffering to allow push-only communication
            volatile int* ptr = reinterpret_cast<volatile int*>(&out);
            uint not_ready = 0xffffffff;
            do {
                uint r1, r2, r3;;
                asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" :
                             "=r"(r1),
                             "=r"(r2),
                             "=r"(r3),
                             "=r"(not_ready)
                             : "l"(ptr) : "memory");
            } while (not_ready != 0);

            // send flit to peer
            asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::
                         "l"(ptr),
                         "r"(flit.uints[0]),
                         "r"(flit.uints[1]),
                         "r"(flit.uints[2]),
                         "r"(flit.uints[3])
                         : "memory");
        }
    }

    // recieve data from peer GPU
    for (int i = thread_id;  i < count;  i += num_threads) {
        // calculate position in buffers
        int c, h, w;
        if (channels_last) {
            const int j = i / NC;
            c = i % NC;
            h = j / NW;
            w = j % NW;
        } else {
            const int j = i / NW;
            w = i % NW;
            c = j / NH;
            h = j % NH;
        }
        int4& in = transfer_in[c*transfer_stride_C + h*transfer_stride_H + w*transfer_stride_W];
        T& out = data_out[c*data_out_stride_C + h*data_out_stride_H + w*data_out_stride_W];

        if (zero) {
	    __zero(&out);
        } else {
            // wait to recieve flit from peer
            Flit flit;
            volatile int* ptr = reinterpret_cast<volatile int*>(&in);
            do {
                asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" :
                             "=r"(flit.uints[0]),
                             "=r"(flit.uints[1]),
                             "=r"(flit.uints[2]),
                             "=r"(flit.uints[3])
                             : "l"(ptr) : "memory");
            } while (flit.uints[3] == 0);
            in = {0, 0, 0, 0};

            // unpack value from flit
            out = flit.payload;
        }
    }
}

template<class T, bool channels_last, bool top_zero, bool btm_zero>
#if __CUDA_ARCH__ >= 700
__launch_bounds__(THREADS_PER_CTA)
#endif
__global__ void push_pull_halos_1d_kernel(
        // top halo,
        const T* tih, int tih_stride_C, int tih_stride_H, int tih_stride_W,     // top input halo (local)
        T* tox,                                                                 // top output transfer buffer (remote peer)
        T* tix,                                                                 // top input transfer buffer (local peer)
        T* toh, int toh_stride_C, int toh_stride_H, int toh_stride_W,           // top output halo (local)
        // btm halo
        const T* bih, int bih_stride_C, int bih_stride_H, int bih_stride_W,     // btm input halo (local)
        T* box,                                                                 // btm output transfer buffer (remote peer)
        T* bix,                                                                 // btm input transfer buffer (local peer)
        T* boh, int boh_stride_C, int boh_stride_H, int boh_stride_W,           // btm output halo (local)
        // dimensions
        int NC, int NH, int NW
        )
{
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int num_threads_per_side = (gridDim.x / 2) * blockDim.x;
    const bool in_top_block = thread_id < num_threads_per_side;
    const int side_thread_id = in_top_block ? thread_id : thread_id - num_threads_per_side;
    if (in_top_block) {
        push_pull_tensor<T,channels_last,top_zero>(
            tih, tih_stride_C, tih_stride_H, tih_stride_W,
            tox,
            tix,
            toh, toh_stride_C, toh_stride_H, toh_stride_W,
            NC, NH, NW,
            side_thread_id, num_threads_per_side);
    } else {
        push_pull_tensor<T,channels_last,btm_zero>(
            bih, bih_stride_C, bih_stride_H, bih_stride_W,
            box,
            bix,
            boh, boh_stride_C, boh_stride_H, boh_stride_W,
            NC, NH, NW,
            side_thread_id, num_threads_per_side);
    }
}

__global__ void delay_kernel(int delay_nanoseconds, int* counter)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // waste time while doing something compiler can't predict, thus preventing it from optimizing away this code.
        int new_counter = 0;
        double elapsed = 0;
        clock_t start = clock();
        do {
            clock_t now = clock();
            elapsed = (double)(now - start)*1e9 / CLOCKS_PER_SEC;
            ++new_counter;
        } while (elapsed < (double)delay_nanoseconds);
        *counter = new_counter;
    }
}

}

namespace apex { namespace contrib { namespace peer_memory {

int64_t allocate_raw(int64_t size)
{
    float* ptr = 0L;
    cudaMalloc(&ptr, size);
    cudaMemset(ptr, 0, size);
    return (int64_t)ptr;
}

void free_raw(int64_t raw)
{
    cudaFree((void*)raw);
}

void zero(int64_t raw, int64_t size)
{
    cudaMemset((void*)raw, 0, size);
}

at::Tensor get_raw_ipc_address(int64_t raw)
{
    cudaIpcMemHandle_t mem_handle;
    CUDACHECK( cudaIpcGetMemHandle(&mem_handle, (void*)raw) );
    const int n = sizeof(cudaIpcMemHandle_t);
    auto address_tensor = torch::empty({n}, torch::dtype(torch::kUInt8));
    auto address_tensor_p = address_tensor.data_ptr<uint8_t>();
    memcpy(address_tensor_p, (uint8_t*)&mem_handle, n);
    return address_tensor;
}

std::vector<int64_t> get_raw_peers(at::Tensor ipc_addresses, int peer_rank, int64_t raw)
{
    int peer_group_size = ipc_addresses.size(0);
    std::vector<int64_t> results(peer_group_size);
    for (int i = 0;  i < peer_group_size;  ++i) {
        if (i != peer_rank) {
            cudaIpcMemHandle_t mem_handle;
            memcpy(&mem_handle, ipc_addresses.index({i}).data_ptr<uint8_t>(), sizeof(cudaIpcMemHandle_t));
            void* p = 0L;
            CUDACHECK( cudaIpcOpenMemHandle((void**)&p, mem_handle, cudaIpcMemLazyEnablePeerAccess) );
            results[i] = (int64_t)p;
        } else {
            results[i] = (int64_t)raw;
        }
    }
    return results;
}

at::Tensor blob_view_half(int64_t raw, std::vector<int64_t> shape, bool channels_last)
{
    return blob_view<at::Half>((at::Half*)raw, shape, torch::dtype(torch::kFloat16).device(torch::kCUDA), channels_last);
}

at::Tensor blob_view_float(int64_t raw, std::vector<int64_t> shape, bool channels_last)
{
    return blob_view<float>((float*)raw, shape, torch::dtype(torch::kFloat32).device(torch::kCUDA), channels_last);
}

at::Tensor blob_view_int(int64_t raw, std::vector<int64_t> shape, bool channels_last)
{
    return blob_view<int>((int*)raw, shape, torch::dtype(torch::kInt32).device(torch::kCUDA), channels_last);
}

void push_pull_halos_1d(
	bool diagnostics,
        bool explicit_nhwc,
        int numSM,                      // number of SMs to use (zero corresponds to all SMs)
	bool top_zero,			// if top halo should be zeroed
        at::Tensor top_in_halo,         // top input halo buffer (in local device memory, sent to top neighbor)
        at::Tensor top_out_transfer,    // top output transfer buffer (in top neighbor peer memory)
	at::Tensor top_in_transfer,	// top input transfer buffer (in local peer memory)
        at::Tensor top_out_halo,        // top output halo buffer (in local device memory, received from top neighbor)
	bool btm_zero,			// if btm halo should be zeroed
        at::Tensor btm_in_halo,         // btm input halo buffer (in local device memory, sent to btm neighbor)
        at::Tensor btm_out_transfer,    // btm output transfer buffer (in btm neighbor peer memory)
	at::Tensor btm_in_transfer,	// btm input transfer buffer (in local peer memory)
        at::Tensor btm_out_halo         // btm output halo buffer (in local device memory, received from btm neighbor)
        )
{
    // basic checks of inputs
    TORCH_CHECK(!(top_zero && btm_zero));
    TORCH_CHECK(top_in_halo.is_cuda());
    TORCH_CHECK(top_out_transfer.is_cuda());
    TORCH_CHECK(top_in_transfer.is_cuda());
    TORCH_CHECK(top_out_halo.is_cuda());
    TORCH_CHECK(btm_in_halo.is_cuda());
    TORCH_CHECK(btm_out_transfer.is_cuda());
    TORCH_CHECK(btm_in_transfer.is_cuda());
    TORCH_CHECK(btm_out_halo.is_cuda());

    // tensor shapes
    int tih_N, tih_C, tih_H, tih_W;
    tensor_shape(top_in_halo, explicit_nhwc, tih_N, tih_C, tih_H, tih_W);
    int toh_N, toh_C, toh_H, toh_W;
    tensor_shape(top_out_halo, explicit_nhwc, toh_N, toh_C, toh_H, toh_W);
    int bih_N, bih_C, bih_H, bih_W;
    tensor_shape(btm_in_halo, explicit_nhwc, bih_N, bih_C, bih_H, bih_W);
    int boh_N, boh_C, boh_H, boh_W;
    tensor_shape(btm_out_halo, explicit_nhwc, boh_N, boh_C, boh_H, boh_W);
    TORCH_CHECK(toh_N == tih_N && tih_N == boh_N && boh_N == bih_N &&
                toh_C == tih_C && tih_C == boh_C && boh_C == bih_C &&
                toh_H == tih_H && tih_H == boh_H && boh_H == bih_H &&
                toh_W == tih_W && tih_W == boh_W && boh_W == bih_W);
    int NN=toh_N, NC=toh_C, NH=toh_H, NW=toh_W;
    if (diagnostics) printf("NN=%d, NC=%d, NH=%d, NW=%d\n",NN,NC,NH,NW);
    TORCH_CHECK(NN == 1);

    // tensor strides
    int tih_stride_N, tih_stride_C, tih_stride_H, tih_stride_W;
    tensor_strides(top_in_halo, explicit_nhwc, tih_stride_N, tih_stride_C, tih_stride_H, tih_stride_W);
    int toh_stride_N, toh_stride_C, toh_stride_H, toh_stride_W;
    tensor_strides(top_out_halo, explicit_nhwc, toh_stride_N, toh_stride_C, toh_stride_H, toh_stride_W);
    int bih_stride_N, bih_stride_C, bih_stride_H, bih_stride_W;
    tensor_strides(btm_in_halo, explicit_nhwc, bih_stride_N, bih_stride_C, bih_stride_H, bih_stride_W);
    int boh_stride_N, boh_stride_C, boh_stride_H, boh_stride_W;
    tensor_strides(btm_out_halo, explicit_nhwc, boh_stride_N, boh_stride_C, boh_stride_H, boh_stride_W);

    // determine if nhwc
    bool is_nhwc = (toh_stride_C == 1);
    if (diagnostics) printf("is_nhwc = %s\n",is_nhwc?"true":"false");

    // peer memory buffers
    int tox_size = top_out_transfer.numel() * top_out_transfer.element_size();
    int tix_size = top_in_transfer.numel() * top_in_transfer.element_size();
    int box_size = btm_out_transfer.numel() * btm_out_transfer.element_size();
    int bix_size = btm_in_transfer.numel() * btm_in_transfer.element_size();
    if (!top_zero) {
        TORCH_CHECK(top_out_transfer.is_contiguous());
        TORCH_CHECK(top_in_transfer.is_contiguous());
    }
    if (!btm_zero) {
        TORCH_CHECK(btm_out_transfer.is_contiguous());
        TORCH_CHECK(btm_in_transfer.is_contiguous());
    }

    // figure out launch parameters
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (numSM <= 0 || numSM > prop.multiProcessorCount) {
      numSM = prop.multiProcessorCount;
    }
    auto current_stream = at::cuda::getCurrentCUDAStream();
    dim3 block(THREADS_PER_CTA,1,1);

    // helper macros to launch templated kernel
#define LAUNCH_PUSH_PULL_HALO_KERNEL_BASE(T, IS_HWC, TOP_ZERO, BTM_ZERO, KERNEL_ARGS, NUM_ELEMENTS) \
    do {                                                                \
        /* require 128b peer memory per element */                      \
        int peer_memory_size = NUM_ELEMENTS * 16;                       \
        if (!TOP_ZERO) {                                                \
            TORCH_CHECK(tox_size >= peer_memory_size && tix_size >= peer_memory_size); \
        }                                                               \
        if (!BTM_ZERO) {                                                \
            TORCH_CHECK(box_size >= peer_memory_size && bix_size >= peer_memory_size); \
        }                                                               \
                                                                        \
        /* launch kernel */                                             \
        int numBlocksPerSm;                                             \
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(                  \
            &numBlocksPerSm,                                            \
            push_pull_halos_1d_kernel<T,IS_HWC,TOP_ZERO,BTM_ZERO>,      \
            THREADS_PER_CTA,                                            \
            0);                                                         \
        dim3 grid(numSM*numBlocksPerSm,1,1);                            \
        if (grid.x % 2 != 0) {                                          \
            /* require even number of blocks (half for top, half for bottom) */ \
            grid.x -= 1;                                                \
        }                                                               \
        if ((grid.x / 2) * block.x > NUM_ELEMENTS) {                    \
            /* only need enough blocks to cover top and bottom halo elements */ \
            grid.x = 2 * ((NUM_ELEMENTS + block.x - 1) / block.x);      \
        }                                                               \
        cudaLaunchCooperativeKernel(                                    \
            (void*)push_pull_halos_1d_kernel<T,IS_HWC,TOP_ZERO,BTM_ZERO>, \
            grid,                                                       \
            block,                                                      \
            KERNEL_ARGS,                                                \
            0,                                                          \
            current_stream);                                            \
    } while (false)
#define LAUNCH_PUSH_PULL_HALO_KERNEL(T, IS_HWC, KERNEL_ARGS, NUM_ELEMENTS) \
    do {                                                                \
        if (top_zero) {                                                 \
            LAUNCH_PUSH_PULL_HALO_KERNEL_BASE(T, IS_HWC, true, false, KERNEL_ARGS, NUM_ELEMENTS); \
        } else if (btm_zero) {                                          \
            LAUNCH_PUSH_PULL_HALO_KERNEL_BASE(T, IS_HWC, false, true, KERNEL_ARGS, NUM_ELEMENTS); \
        } else {                                                        \
            LAUNCH_PUSH_PULL_HALO_KERNEL_BASE(T, IS_HWC, false, false, KERNEL_ARGS, NUM_ELEMENTS); \
        }                                                               \
    } while (false)

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, top_out_halo.scalar_type(), "push_pull_halos_1d_kernel", [&]{
	if (diagnostics) printf("size(scalar_t) = %ld\n",sizeof(scalar_t));
        scalar_t* tih_p = top_inp_halo.data_ptr<scalar_t>();
        int4* tox_p = top_out_tx.data_ptr<int4>();
        int4* tix_p = top_inp_tx.data_ptr<int4>();
        scalar_t* toh_p = top_out_halo.data_ptr<scalar_t>();
        scalar_t* bih_p = btm_inp_halo.data_ptr<scalar_t>();
        int4* box_p = btm_out_tx.data_ptr<int4>();
        int4* bix_p = btm_inp_tx.data_ptr<int4>();
        scalar_t* boh_p = btm_out_halo.data_ptr<scalar_t>();
        if (diagnostics) printf("waypoint1\n");

        // do int2 vector loads if channel count permits
        int elem_size_in_bytes = toh_C * sizeof(scalar_t);
        int elem_size_in_int2 = (elem_size_in_bytes / 8);
        if (diagnostics) printf("elem_size_in_bytes = %d, elem_size_in_int2 = %d\n",elem_size_in_bytes,elem_size_in_int2);
        if (is_nhwc && elem_size_in_int2*8 == elem_size_in_bytes) {
            // can do int2 transfers
            int divisor = 8 / sizeof(scalar_t);
            if (diagnostics) printf("CAN DO INT2 :: divisor = %d\n",divisor);
            toh_stride_N /= divisor;   toh_stride_H /= divisor;    toh_stride_W /= divisor;
            tox_stride_N /= divisor;   tox_stride_H /= divisor;    tox_stride_W /= divisor;
            tix_stride_N /= divisor;   tix_stride_H /= divisor;    tix_stride_W /= divisor;
            tih_stride_N /= divisor;   tih_stride_H /= divisor;    tih_stride_W /= divisor;
            boh_stride_N /= divisor;   boh_stride_H /= divisor;    boh_stride_W /= divisor;
            box_stride_N /= divisor;   box_stride_H /= divisor;    box_stride_W /= divisor;
            bix_stride_N /= divisor;   bix_stride_H /= divisor;    bix_stride_W /= divisor;
            bih_stride_N /= divisor;   bih_stride_H /= divisor;    bih_stride_W /= divisor;
            NC /= divisor;
            if (diagnostics) {
                printf("divisor=%d\n",divisor);
                printf("tih_stride :: N=%d, C=%d, H=%d, W=%d\n",tih_stride_N,tih_stride_C,tih_stride_H,tih_stride_W);
                printf("toh_stride :: N=%d, C=%d, H=%d, W=%d\n",toh_stride_N,toh_stride_C,toh_stride_H,toh_stride_W);
                printf("bih_stride :: N=%d, C=%d, H=%d, W=%d\n",bih_stride_N,bih_stride_C,bih_stride_H,bih_stride_W);
                printf("boh_stride :: N=%d, C=%d, H=%d, W=%d\n",boh_stride_N,boh_stride_C,boh_stride_H,boh_stride_W);
                printf("NC=%d, NH=%d, NW=%d\n",NC,NH,NW);
            }
            void *kernel_args[] = {
                (int2**)&tih_p, &tih_stride_C, &tih_stride_H, &tih_stride_W,
                &tox_p,
                &tix_p,
                (int2**)&toh_p, &toh_stride_C, &toh_stride_H, &toh_stride_W,
                (int2**)&bih_p, &bih_stride_C, &bih_stride_H, &bih_stride_W,
                &box_p,
                &bix_p,
                (int2**)&boh_p, &boh_stride_C, &boh_stride_H, &boh_stride_W,
                &NC, &NH, &NW
            };
            int num_elem = NC*NH*NW;
            LAUNCH_PUSH_PULL_HALO_KERNEL(int2, true, kernel_args, num_elem);
        } else {
            // cannot do int2 transfers
            if (diagnostics) printf("CAN NOT DO INT2\n");
            void *kernel_args[] = {
		&tih_p, &tih_stride_C, &tih_stride_H, &tih_stride_W,
                &tox_p,
                &tix_p,
                &toh_p, &toh_stride_C, &toh_stride_H, &toh_stride_W,
                &bih_p, &bih_stride_C, &bih_stride_H, &bih_stride_W,
                &box_p,
                &bix_p,
                &boh_p, &boh_stride_C, &boh_stride_H, &boh_stride_W,
                &NC, &NH, &NW
            };
            int num_elem = NC*NH*NW;
            if (is_nhwc) {
                LAUNCH_PUSH_PULL_HALO_KERNEL(scalar_t, true, kernel_args, num_elem);
            } else {
                LAUNCH_PUSH_PULL_HALO_KERNEL(scalar_t, false, kernel_args, num_elem);
            }
        }
    } );

#undef LAUNCH_PUSH_PULL_HALO_KERNEL_BASE
#undef LAUNCH_PUSH_PULL_HALO_KERNEL
}

} } }
