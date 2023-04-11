# Script for fetching the git hash with potential dirty status

function(git_version)
    execute_process(
        COMMAND git describe --always --dirty
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        OUTPUT_VARIABLE GIT_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    return(PROPAGATE GIT_VERSION)
endfunction()