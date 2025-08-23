The latest version of asr-webservice has added support for the RTX 5090 GPU. By recompiling the source code, we have added support for torch2.7+cuda128. Now, it can efficiently provide transcription services for applications like Speaker using the GPU!

You can directly pull the pre-built image I have packaged using the following command:

docker pull crpi-n9jif4z5nex2rnkd.cn-hangzhou.personal.cr.aliyuncs.com/docker_2025-images/whisper-asr-webservice_for_5090:latest

为asr-webservice的最新版本添加RTX 5090显卡支持，通过对源代码重新编译，添加了torch2.7+cuda128，现在可以使用GPU来高效地为Speaker等应用提供转写服务啦！

直接通过以下命令拉取我打包好的镜像即可：

docker pull crpi-n9jif4z5nex2rnkd.cn-hangzhou.personal.cr.aliyuncs.com/docker_2025-images/whisper-asr-webservice_for_5090:latest
