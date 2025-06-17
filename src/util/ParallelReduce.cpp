#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <memory>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "ParallelReduce.h"

namespace
{
	// true to use one reduction CPU thread per core, like 8 threads on Ryzen 7 8700G
	// false to use one reduction thread per hardware thread, 16 threads on that Ryzen
	constexpr bool oneThreadPerCore = true;

	inline BOOL getCoresInfo( void* rdi, DWORD& len ) noexcept
	{
		return GetLogicalProcessorInformationEx( RelationProcessorCore, (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)rdi, &len );
	}

	inline uint32_t computePhysicalCoreCount() noexcept
	{
		DWORD len = 0;
		getCoresInfo( nullptr, len );
		if( GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0 )
			return 0;

		std::unique_ptr<uint8_t[]> buffer;
		try
		{
			buffer = std::make_unique<uint8_t[]>( len );
		}
		catch( const std::bad_alloc& )
		{
			return 0;
		}

		if( !getCoresInfo( buffer.get(), len ) )
			return 0;

		const uint8_t* rsi = buffer.get();
		const uint8_t* const rsiEnd = rsi + len;
		uint32_t cores = 0;
		while( rsi < rsiEnd )
		{
			const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* info = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)rsi;
			if( info->Relationship == RelationProcessorCore )
				cores++;
			rsi += info->Size;
		}
		return cores;
	}

	inline uint32_t hardwareThreads() noexcept
	{
		SYSTEM_INFO sysInfo;
		GetSystemInfo( &sysInfo );
		return sysInfo.dwNumberOfProcessors;
	}

	uint32_t computeHardwareConcurrency() noexcept
	{
#ifdef NDEBUG
		uint32_t res = 0;
		if constexpr( oneThreadPerCore )
			res = computePhysicalCoreCount();
		// Failing that, use count of hardware threads
		if( 0 == res )
			res = hardwareThreads();
		return res;
#else
		// In debug builds initialize s_hwConcurrency with 3 to simplify debugging
		return 3;
#endif
	}

	static const uint32_t s_hwConcurrency = computeHardwareConcurrency();

	using namespace Parallel;
}

ReductionJob::ReductionJob( size_t alignEntry ):
	countWorkers( s_hwConcurrency ),
	cbBufferAlign( alignEntry )
{
	m_work = CreateThreadpoolWork( &workCallbackStatic, this, nullptr );
	if( nullptr == m_work )
	{
		HRESULT hr = HRESULT_FROM_WIN32( GetLastError() );
		throw std::exception( "CreateThreadpoolWork", hr );
	}
}

bool ReductionJob::allocateBuffer( size_t cbEntry ) noexcept
{
	if( cbEntry == cbReductionEntry )
		return true;	// Already of the correct size
	assert( 0 == ( cbEntry % cbBufferAlign ) );

	// If necessary, free the old buffer
	if( nullptr != reductionBuffer )
	{
		_aligned_free( reductionBuffer );
		reductionBuffer = nullptr;
	}
	cbReductionEntry = 0;

	// Allocate the new buffer, align up to a multiple of 64 bytes
	// This enables faster version of zeroMemory method
	size_t cbBuffer = cbEntry * countWorkers;
	cbBuffer = ( cbBuffer + 63 ) & (ptrdiff_t)( -64 );
	reductionBuffer = _aligned_malloc( cbBuffer, cbBufferAlign );
	if( nullptr == reductionBuffer )
		return false;

	cbReductionEntry = cbEntry;
	return true;
}

uint8_t* ReductionJob::getThreadBuffer( size_t idxThread ) const noexcept
{
	assert( idxThread < countWorkers );
	return getBuffer() + ( idxThread * cbReductionEntry );
}

ReductionJob::~ReductionJob()
{
	if( nullptr != m_work )
	{
		WaitForThreadpoolWorkCallbacks( m_work, TRUE );
		CloseThreadpoolWork( m_work );
		m_work = nullptr;
	}

	if( nullptr != reductionBuffer )
	{
		_aligned_free( reductionBuffer );
		reductionBuffer = nullptr;
	}
}

void ReductionJob::zeroMemory() noexcept
{
	uint8_t* rdi = (uint8_t*)reductionBuffer;
	uint8_t* const rdiEnd = rdi + ( countWorkers * cbReductionEntry );
	const __m256i zero = _mm256_setzero_si256();
	for( ; rdi < rdiEnd; rdi += 64 )
	{
		_mm256_storeu_si256( ( __m256i* )( rdi ), zero );
		_mm256_storeu_si256( ( __m256i* )( rdi + 32 ), zero );
	}
}

void ReductionJob::dispatch( pfnPoolCallback callback, void* context, size_t countJobs ) noexcept
{
	assert( nullptr != callback && 0 != countJobs );
	assert( nullptr == pfn && nullptr != m_work );

	pfn = callback;
	pv = context;
	nextJob = -1;
	nextThread = -1;
	this->countJobs = countJobs;

	// Launch up to ( countWorkers - 1 ) background jobs
	const size_t workers = std::min( countJobs, countWorkers );
	for( size_t i = 1; i < workers; i++ )
		SubmitThreadpoolWork( m_work );

	// Run the same job on the current thread as well
	workCallbackStatic( nullptr, this, m_work );

	// Wait for the background workers to complete
	if( workers > 1 )
		WaitForThreadpoolWorkCallbacks( m_work, FALSE );

	// Clear the fields of this class
	pfn = nullptr;
	pv = nullptr;
}

inline void ReductionJob::workCallback() noexcept
{
	const pfnPoolCallback pfn = this->pfn;
	void* const pv = this->pv;
	const size_t length = countJobs;
	// Interlocked increment to find index of the current thread
	const size_t threadIdx = InterlockedIncrement64( &nextThread );
	uint8_t* const rdi = getThreadBuffer( threadIdx );

	while( true )
	{
		// Interlocked increment to find index of the current task
		const size_t idx = InterlockedIncrement64( &nextJob );

		// Check for completion
		if( idx < length )
			pfn( pv, idx, threadIdx, rdi );
		else
			return;
	}
}

void __declspec( noinline ) __stdcall ReductionJob::workCallbackStatic( PTP_CALLBACK_INSTANCE instance, void* context, PTP_WORK work ) noexcept
{
	( (ReductionJob*)( context ) )->workCallback();
}