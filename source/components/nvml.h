#pragma once

// *************** FOR ERROR CHECKING *******************
#ifndef IA_NVML_CALL
#define IA_NVML_CALL( call )                                                                                           \
    {                                                                                                                  \ 
        auto status = static_cast<nvmlReturn_t>( call );                                                               \
        if ( status != NVML_SUCCESS )                                                                                  \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA NVML call \"%s\" in line %d of file %s failed "                                      \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     ia_nvml::pfn_nvmlErrorString( status ),                                                                    \
                     status );                                                                                         \
    }
#endif  // NVML_RT_CALL
// *************** FOR ERROR CHECKING *******************

namespace ia_nvml {
void query_nvml_pointers();
}
