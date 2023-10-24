# ZeRO-Offload++

This folder contains examples that demonstrate how to use the new ZeRO-Offload++ features. 

Now ZeRO-Offload++ supports **Twin-Offload** feature.

## Twin-offload

Instead of all-or-nothing offloading strategy, **Twin-Offload** allows a portion of data to run on CPU and the other part on GPU simultaneously. Thus, we not only mitigate the memory pressure on GPU side by offloading data to CPU, but also utilize CPU computation resources more efficiently. 

## How to use



## On-going optimizations

* Removing uncessary D2H memcpy in ZeRO-offload
* On-the-fly fp16 to fp32 data casting inside CPUAdam
