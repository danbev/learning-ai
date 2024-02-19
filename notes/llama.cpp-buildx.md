## Docker buildx on llama.cpp
Docker buildx need to be installed by donwloading the latest version from
https://github.com/docker/buildx/releases/tag/v0.12.1. Copy and rename the
binary file as ~/.docker/cli-plugins/docker-buildx. Then run `docker buildx` to
verifiy the installation:
```console
(llava-venv) $ docker buildx

Usage:  docker buildx [OPTIONS] COMMAND

Extended build capabilities with BuildKit
...
```

The docker files for llamaa.cpp are in the .devops directory.

The motivation for looking into this is that a PR that I created failed on one
of these builds:
```error
246.1 /usr/include/c++/11/type_traits:79:52: error: redefinition of 'constexpr const _Tp std::integral_constant<_Tp, __v>::value'
246.1    79 |   template<typename _Tp, _Tp __v>
246.1       |                                                    ^                           
246.1 /usr/include/c++/11/type_traits:67:29: note: 'constexpr const _Tp value' previously declared here
246.1    67 |       static constexpr _Tp                  value = __v;
246.1       |                             ^~~~~
246.9 make: *** [Makefile:447: ggml-cuda.o] Error 1
```

And the command that caused this failure was:
```console
/usr/bin/docker buildx build --file .devops/main-cuda.Dockerfile --iidfile /tmp/docker-actions-toolkit-ADmprz/iidfile --platform linux/amd64 --provenance mode=max,builder-id=https://github.com/ggerganov/llama.cpp/actions/runs/7955662322 --tag ghcr.io/ggerganov/llama.cpp:light-cuda --tag ghcr.io/ggerganov/llama.cpp:light-cuda--b1-5fdca56 --metadata-file /tmp/docker-actions-toolkit-ADmprz/metadata-file .
```

I'm trying to run this locally using after commenting out everything in the
DockerFile after the make command:
```console
(llava-venv) $ docker buildx build --file .devops/main-cuda.Dockerfile .
...

253.9 /usr/include/c++/11/type_traits:79:52: error: redefinition of 'constexpr const _Tp std::integral_constant<_Tp, __v>::value'
253.9    79 |   template<typename _Tp, _Tp __v>
253.9       |                                                    ^                           
253.9 /usr/include/c++/11/type_traits:67:29: note: 'constexpr const _Tp value' previously declared here
253.9    67 |       static constexpr _Tp                  value = __v;
253.9       |                             ^~~~~
254.8 make: *** [Makefile:447: ggml-cuda.o] Error 1
------
main-cuda.Dockerfile:26
--------------------
  24 |     ENV LLAMA_CUBLAS=1
  25 |     
  26 | >>> RUN make
  27 |     
  28 |     #FROM ${BASE_CUDA_RUN_CONTAINER} as runtime
--------------------
ERROR: failed to solve: process "/bin/sh -c make" did not complete successfully: exit code: 2
...
Great, so we can run this locally. Now, lets print out the verison of gcc that
is being used. So I added a RUN gcc --version command to the DockerFile but it
just swooshed by. I read that setting export BUILDKIT_PROGRESS=plain might allow
the output to be displayed so lets try that.
```
$ env BUILDKIT_PROGRESS=plain docker buildx build --file .devops/main-cuda.Dockerfile .
...
#9 [5/6] RUN gcc --version
#9 0.614 gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
#9 0.614 Copyright (C) 2021 Free Software Foundation, Inc.
#9 0.614 This is free software; see the source for copying conditions.  There is NO
#9 0.614 warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#9 0.614 
#9 DONE 0.6s
```
So this is usig gcc 11.4.0 and the cuda version is 11.7.1.
Hmm, acutally the error is a c++ error so lets just check g++ as well to be sure
about this the version:
```console
10 [6/7] RUN g++ --version
#10 0.235 g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
#10 0.235 Copyright (C) 2021 Free Software Foundation, Inc.
#10 0.235 This is free software; see the source for copying conditions.  There is NO
#10 0.235 warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#10 0.235 
```
Alright, so we know we are getting this error with g++ 11.4.0. 
Could this be related to that an update of build-essentials that brought in a
newer version of gcc/g++. For example if it was using 10.x before then perhaps
11 introduced this error. I'll try installing gcc-10 and g++-10 and see if that
allows the build to pass and perhaps this could be a workaround until a proper
fix is in place. That worked an allowd the build to pass without the error,
though I still don't the root cause of this.
