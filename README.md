### DOCA Packet Processing

This repo was used for our work on GPU-based packet processing:

Van Hauwaert, Romain ; Vanliefde, Maxime. GPU-based Packet Processing.  Ecole polytechnique de Louvain, Université catholique de Louvain, 2024. Prom. : Barbette, Tom. http://hdl.handle.net/2078.1/thesis:45872

This application is a modified version of the [Ethernet Simple Receive](https://docs.nvidia.com/doca/sdk/doca+gpunetio/index.html#src-2845250315_id-.DOCAGPUNetIOv2.8.0-EthernetSimpleReceive) sample from GPUNetIO.
It processes packets on the GPU by directly sending the received packets from the interface to the GPU, meaning the CPU is not used to process these packets.
Three workloads are implemented: an EtherMirror, an IP Lookup and a CRC.
The packets are then sent back to measure the application's performance.
Our measures were made using the [GPU NPF](https://github.com/OnyxDurga/gpu-npf) repo.

### Build

To build the application, use the following commands:

```
meson build
ninja -C build
```

### Run

To run the application, use the following command:
```
sudo ./build/doca_gpunetio_simple_receive -n nic -g gpu -q queues -w workload -b batching
```

Where nic is the PCIe address of the NIC. gpu is the PCIe address of the GPU. queues is the number of queues, between 0 and 8. Workload is the type of workload to process on the GPU, 0 is only the EtherMirror, 1 is EtherMirror and IP Lookup, 2 is EtherMirror and CRC.
Finally, batching is the number of packets to receive at a time on the GPU, between 0 and 2048. In our testing, this parameter did not seem to be working well.
