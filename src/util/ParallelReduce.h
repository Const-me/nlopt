#pragma once
#include <cstdint>

// Forward declarations because we don't want to include <windows.h> in this header
typedef struct _TP_WORK* PTP_WORK;
typedef struct _TP_CALLBACK_INSTANCE* PTP_CALLBACK_INSTANCE;

namespace Parallel
{
	// Type-agnostic implementation of the parallel reduction algorithm built on top of Windows thread pool
	struct ReductionJob
	{
		// Initialize thread pool job
		ReductionJob( size_t bufferAlign );
		~ReductionJob();
		ReductionJob( const ReductionJob& ) = delete;

		// allocate buffer for the reduction; the buffer is on the heap
		bool allocateBuffer( size_t cbEntry ) noexcept;

		using pfnPoolCallback = void( * )( void* pvContext, size_t idxJob, size_t idxThread, uint8_t* rdi );

		// Fill the entire reduction buffer with zero bytes
		void zeroMemory() noexcept;

		// Dispatch parallel work on thread pool, wait for completion
		void dispatch( pfnPoolCallback callback, void* context, size_t countJobs ) noexcept;

		uint8_t* getBuffer() const noexcept { return (uint8_t*)reductionBuffer; }

		const size_t countWorkers;

		size_t cbEntry() const noexcept { return cbReductionEntry; }
	private:
		uint8_t* getThreadBuffer( size_t idxThread ) const noexcept;

		PTP_WORK m_work;
		const size_t cbBufferAlign;
		void* reductionBuffer = nullptr;
		size_t cbReductionEntry = 0;

		pfnPoolCallback pfn = nullptr;
		void* pv = nullptr;

		static void __stdcall workCallbackStatic( PTP_CALLBACK_INSTANCE instance, void* context, PTP_WORK work ) noexcept;

		void workCallback() noexcept;

		size_t countJobs = 0;

		// Now the volatile fields, we want them on separate cache line
		alignas( 64 ) volatile int64_t nextJob = -1;
		volatile int64_t nextThread = -1;
	};
}

// Parallel reduction algorithm built on top of Windows thread pool
template<class Context>
class ParallelReduce
{
	Parallel::ReductionJob impl;

public:
	// Align the reduction buffer by cache lines = 64 bytes
	ParallelReduce(): impl( 64 ) { };
	~ParallelReduce() = default;

	// Allocate thread-local reduction buffer for the worker threads
	// If already allocated the same length, do nothing and report success
	bool allocateBuffer( size_t doubles ) noexcept
	{
		// Round up to multiple of 8 entries = 64 bytes = cache line
		doubles = ( doubles + 7 ) & (ptrdiff_t)-8;

		// Scale the length from FP64 elements into bytes, and call the implementation method
		const size_t bytes = doubles * sizeof( double );
		return impl.allocateBuffer( bytes );
	}

	using pfnPoolCallback = void( * )( Context* const ctx, const size_t idxJob, const size_t idxThread, double* const rdi );

	// Fill the entire reduction buffer with zero bytes
	void zeroMemory() noexcept { impl.zeroMemory(); }

	// Dispatch parallel work on thread pool, wait for completion
	// The implementation uses work-stealing, each thread pulls tasks with integer atomic increments
	// Tasks [ 0 .. countJobs - 1 ] are dispatched in dynamic order for load balancing across CPU cores.
	void dispatch( pfnPoolCallback callback, Context* context, size_t countJobs ) noexcept
	{
		impl.dispatch( (Parallel::ReductionJob::pfnPoolCallback)callback, context, countJobs );
	}

	size_t countWorkers() const noexcept
	{
		return impl.countWorkers;
	}

	// Length of the thread-local reduction buffer for each thread.
	// The returned number is expressed in FP64 elements, and includes padding.
	size_t threadBufferSize() const noexcept
	{
		return impl.cbEntry() / sizeof( double );
	}

	// Pointer to the start of the internal reduction buffer
	// Buffer length in elements is equal to the product of threadBufferSize() and countWorkers()
	double* buffer() const noexcept
	{
		return (double*)impl.getBuffer();
	}
};