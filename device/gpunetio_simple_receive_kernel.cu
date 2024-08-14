/*
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of NVIDIA CORPORATION &
 * AFFILIATES (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 *
 */

#include <doca_gpunetio_dev_buf.cuh>
#include <doca_gpunetio_dev_eth_txq.cuh>
#include <doca_gpunetio_dev_eth_rxq.cuh>
#include <doca_log.h>
#include <rte_ip.h>
#include <rte_ether.h>

#include "gpunetio_common.h"

DOCA_LOG_REGISTER(GPU_SEND_WAIT_TIME::KERNEL);

__device__ bool matches_prefix(uint32_t addr1, uint32_t addr2, uint32_t mask)
{
    return ((addr1 ^ addr2) & mask) == 0;
}

__device__ uint32_t lookup_entry(uint32_t a, ip_vector_t *ip_list, uint32_t len) 
{
    uint64_t found = 0;
    for (int i = 0; i < len - 1; i++) {
        ip_vector_t r = ip_list[i];
	    bool b = matches_prefix(a, r.addr, r.mask);
        if (b) found = i;

	}
    return found;
}

__device__ uint32_t lookup_route(uint32_t a, ip_vector_t *ip_list, uint32_t len) 
{
    int ei = lookup_entry(a, ip_list, len);

    if (ei >= 0) {
	return ip_list[ei].port;
    } else
	return 0;
}

__global__ void receive_packets(struct doca_gpu_eth_rxq *eth_rxq_gpu0, struct doca_gpu_eth_rxq *eth_rxq_gpu1, struct doca_gpu_eth_rxq *eth_rxq_gpu2, struct doca_gpu_eth_rxq *eth_rxq_gpu3,
								struct doca_gpu_eth_rxq *eth_rxq_gpu4, struct doca_gpu_eth_rxq *eth_rxq_gpu5, struct doca_gpu_eth_rxq *eth_rxq_gpu6, struct doca_gpu_eth_rxq *eth_rxq_gpu7, 
								uint32_t *exit_cond, 
								struct doca_gpu_eth_txq *txq0, struct doca_gpu_eth_txq *txq1, struct doca_gpu_eth_txq *txq2, struct doca_gpu_eth_txq *txq3,
								struct doca_gpu_eth_txq *txq4, struct doca_gpu_eth_txq *txq5, struct doca_gpu_eth_txq *txq6, struct doca_gpu_eth_txq *txq7,
								int numq,
								uint8_t workload,
								ip_vector_t* _ip_list_gpu, uint32_t _ip_list_len,
								uint32_t *crc_table,
								uint32_t max_rx_num)
{
	__shared__ uint32_t rx_pkt_num;
	__shared__ uint64_t rx_buf_idx;

	doca_error_t ret;
	struct doca_gpu_eth_rxq *rxq = NULL;
	struct doca_gpu_eth_txq *txq = NULL;
	struct doca_gpu_buf *buf_ptr = NULL;
	uintptr_t buf_addr;
	uint64_t buf_idx;
	uint16_t nbytes;

	struct rte_ether_hdr *eth;
    struct rte_ipv4_hdr *ipv4;

	uint32_t lane_id = threadIdx.x % CUDA_BLOCK_THREADS;
	uint32_t warp_id = threadIdx.x / CUDA_BLOCK_THREADS;

	// printf("hello from thread %d, block: %d\n", threadIdx.x, blockIdx.x);
	if (blockIdx.x == 0) {
		rxq = eth_rxq_gpu0;
		txq = txq0;
	} else if (blockIdx.x == 1) {
		rxq = eth_rxq_gpu1;
		txq = txq1;
	} else if (blockIdx.x == 2) {
		rxq = eth_rxq_gpu2;
		txq = txq2;
	} else if (blockIdx.x == 3) {
		rxq = eth_rxq_gpu3;
		txq = txq3;
	} else if (blockIdx.x == 4) {
		rxq = eth_rxq_gpu4;
		txq = txq4;
	} else if (blockIdx.x == 5) {
		rxq = eth_rxq_gpu5;
		txq = txq5;
	} else if (blockIdx.x == 6) {
		rxq = eth_rxq_gpu6;
		txq = txq6;
	} else if (blockIdx.x == 7) {
		rxq = eth_rxq_gpu7;
		txq = txq7;
	}
	else
		return;

	if (warp_id > 0)
		return;

	while (DOCA_GPUNETIO_VOLATILE(*exit_cond) == 0) {
		// printf("batch: %d\n", max_rx_num);
		ret = doca_gpu_dev_eth_rxq_receive_block(rxq, max_rx_num, 0, &rx_pkt_num, &rx_buf_idx);
		/* If any thread returns receive error, the whole execution stops */
		if (ret != DOCA_SUCCESS) {
			if (threadIdx.x == 0) {
				/*
				 * printf in CUDA kernel may be a good idea only to report critical errors or debugging.
				 * If application prints this message on the console, something bad happened and
				 * applications needs to exit
				 */
				// printf("Receive UDP kernel error %d rxpkts %d error %d\n", ret, rx_pkt_num, ret);
				DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
			}
			break;
		}

		if (rx_pkt_num == 0)
			continue;

		// printf("recv %d pkts in %d thread in block %d\n", rx_pkt_num, threadIdx.x, blockIdx.x);

		buf_idx = threadIdx.x;
		while (buf_idx < rx_pkt_num) {
			ret = doca_gpu_dev_eth_rxq_get_buf(rxq, rx_buf_idx + buf_idx, &buf_ptr);
			if (ret != DOCA_SUCCESS) {
				printf("UDP Error %d doca_gpu_dev_eth_rxq_get_buf thread %d\n", ret, threadIdx.x);
				DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
				break;
			}

			ret = doca_gpu_dev_buf_get_addr(buf_ptr, &buf_addr);
			if (ret != DOCA_SUCCESS) {
				printf("UDP Error %d doca_gpu_dev_eth_rxq_get_buf thread %d\n", ret, threadIdx.x);
				DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
				break;
			}

			uint8_t *src, *dst, tmp[6];
			size_t size;


			eth = (struct rte_ether_hdr *)(((uint8_t *) (buf_addr)));
        	ipv4 = (struct rte_ipv4_hdr *)((char *)eth + sizeof(struct rte_ether_hdr));

			// printf("workload: %d\n", workload);
			// printf("ip size: %d\n", sizeof(struct rte_ipv4_hdr));

			// // printf("total size: %d\n", ipv4->total_length);
			// // printf("total size: %d\n", ipv4->dst_addr);
			// // printf("total size: %d\n", (uint16_t *)(((uint8_t *)buf_addr)[14+1]));

			// printf("total size: %d\n", BE_TO_CPU_16(ipv4->total_length));
			nbytes = BE_TO_CPU_16(ipv4->total_length);


			src = (uint8_t *) (&eth->src_addr);
			dst = (uint8_t *) (&eth->dst_addr);
					
			// printf("Thread %d in block %d received UDP packet with Eth src %02x:%02x:%02x:%02x:%02x:%02x - Eth dst %02x:%02x:%02x:%02x:%02x:%02x\n",
			// 	threadIdx.x, blockIdx.x,
			// 	((uint8_t *)buf_addr)[0], ((uint8_t *)buf_addr)[1], ((uint8_t *)buf_addr)[2], ((uint8_t *)buf_addr)[3], ((uint8_t *)buf_addr)[4], ((uint8_t *)buf_addr)[5],
			// 	((uint8_t *)buf_addr)[6], ((uint8_t *)buf_addr)[7], ((uint8_t *)buf_addr)[8], ((uint8_t *)buf_addr)[9], ((uint8_t *)buf_addr)[10], ((uint8_t *)buf_addr)[11]
			// );

			// for (int i = 0; i < 60; i++) {
			// 	if (i==14) printf("\n");
			// 	if (i==(14+sizeof(rte_ipv4_hdr))) printf("\n");
			// 	printf("%02x ", ((uint8_t *)buf_addr)[i]);
			// }

			// if (workload == 0) {
			uint8_t j;
			for (j = 0; j < 6; j++) tmp[j] = src[j];
			for (j = 0; j < 6; j++) src[j] = dst[j];
			for (j = 0; j < 6; j++) dst[j] = tmp[j];
			// }

			if (workload == 1) {
				uint32_t port = lookup_route((uint32_t) ipv4->dst_addr, _ip_list_gpu, _ip_list_len);
			}

			else if (workload == 2) {

				size = nbytes + sizeof(rte_ether_hdr);
				
				int i, j;
				uint32_t crc_accum = 0xffffffff;
				char *data = (char *) buf_addr;
				for (j = 0;  j < size;  j++ ) {
					i = ( (uint32_t) ( crc_accum >> 24) ^ *data++ ) & 0xff;
					crc_accum = ( crc_accum << 8 ) ^ crc_table[i];
				}

				ipv4->hdr_checksum = crc_accum;
			}

			// printf("Thread %d received UDP packet with Eth src %02x:%02x:%02x:%02x:%02x:%02x - Eth dst %02x:%02x:%02x:%02x:%02x:%02x\n",
			// 	threadIdx.x,
			// 	((uint8_t *)buf_addr)[0], ((uint8_t *)buf_addr)[1], ((uint8_t *)buf_addr)[2], ((uint8_t *)buf_addr)[3], ((uint8_t *)buf_addr)[4], ((uint8_t *)buf_addr)[5],
			// 	((uint8_t *)buf_addr)[6], ((uint8_t *)buf_addr)[7], ((uint8_t *)buf_addr)[8], ((uint8_t *)buf_addr)[9], ((uint8_t *)buf_addr)[10], ((uint8_t *)buf_addr)[11]
			// );
			
			/* Add packet processing function here. */
			// printf("nbytes: %d\n", nbytes);
			doca_gpu_dev_eth_txq_send_enqueue_strong(txq, buf_ptr, nbytes + sizeof(rte_ether_hdr), DOCA_GPU_SEND_FLAG_NONE);

			buf_idx += blockDim.x;
		}
		__syncthreads();

		if (lane_id == 0) {
			// printf("sending pkts in block %d\n", blockIdx.x);
			doca_gpu_dev_eth_txq_commit_strong(txq);
			doca_gpu_dev_eth_txq_push(txq);
		}

		__syncthreads();
	}
}

extern "C" {

doca_error_t kernel_receive_packets(cudaStream_t stream, struct rxq_queue *rxq, uint32_t *gpu_exit_condition, uint8_t workload, ip_vector_t* _ip_list_gpu, uint32_t _ip_list_len, uint32_t *crc_table, uint32_t max_rx_num)
{
	cudaError_t result = cudaSuccess;

	if (rxq == NULL || rxq->numq == 0 || rxq->numq > MAX_QUEUES || gpu_exit_condition == NULL) {
		DOCA_LOG_ERR("kernel_receive_icmp invalid input values");
		return DOCA_ERROR_INVALID_VALUE;
	}

	/* Check no previous CUDA errors */
	result = cudaGetLastError();
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	receive_packets<<<rxq->numq, 32, 0, stream>>>(	rxq->eth_rxq_gpu[0], rxq->eth_rxq_gpu[1], rxq->eth_rxq_gpu[2], rxq->eth_rxq_gpu[3], \
													rxq->eth_rxq_gpu[4], rxq->eth_rxq_gpu[5], rxq->eth_rxq_gpu[6], rxq->eth_rxq_gpu[7], \
													gpu_exit_condition, \
													rxq->eth_txq_gpu[0], rxq->eth_txq_gpu[1], rxq->eth_txq_gpu[2], rxq->eth_txq_gpu[3], \
													rxq->eth_txq_gpu[4], rxq->eth_txq_gpu[5], rxq->eth_txq_gpu[6], rxq->eth_txq_gpu[7], \
													rxq->numq,
													workload,
													_ip_list_gpu, _ip_list_len,
													crc_table,
													max_rx_num);
	result = cudaGetLastError();
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	return DOCA_SUCCESS;
}

} /* extern C */
