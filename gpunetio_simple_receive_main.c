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

#include <doca_argp.h>
#include <doca_log.h>

#include "gpunetio_common.h"

DOCA_LOG_REGISTER(GPU_SEND_WAIT_TIME::MAIN);

/*
 * Get GPU PCIe address input.
 *
 * @param [in]: Command line parameter
 * @config [in]: Application configuration structure
 * @return: DOCA_SUCCESS on success and DOCA_ERROR_INVALID_VALUE otherwise
 */
static doca_error_t
gpu_pci_address_callback(void *param, void *config)
{
	struct sample_send_wait_cfg *sample_cfg = (struct sample_send_wait_cfg *)config;
	char *pci_address = (char *)param;
	size_t len;

	len = strnlen(pci_address, MAX_PCI_ADDRESS_LEN);
	if (len >= MAX_PCI_ADDRESS_LEN) {
		DOCA_LOG_ERR("PCI address too long. Max %d", MAX_PCI_ADDRESS_LEN - 1);
		return DOCA_ERROR_INVALID_VALUE;
	}

	strncpy(sample_cfg->gpu_pcie_addr, pci_address, len + 1);

	return DOCA_SUCCESS;
}

/*
 * Get NIC PCIe address input.
 *
 * @param [in]: Command line parameter
 * @config [in]: Application configuration structure
 * @return: DOCA_SUCCESS on success and DOCA_ERROR_INVALID_VALUE otherwise
 */
static doca_error_t
nic_pci_address_callback(void *param, void *config)
{
	struct sample_send_wait_cfg *sample_cfg = (struct sample_send_wait_cfg *)config;
	char *pci_address = (char *)param;
	size_t len;

	len = strnlen(pci_address, MAX_PCI_ADDRESS_LEN);
	if (len >= MAX_PCI_ADDRESS_LEN) {
		DOCA_LOG_ERR("PCI address too long. Max %d", MAX_PCI_ADDRESS_LEN - 1);
		return DOCA_ERROR_INVALID_VALUE;
	}

	strncpy(sample_cfg->nic_pcie_addr, pci_address, len + 1);

	return DOCA_SUCCESS;
}

/*
 * Get GPU receive queue number.
 *
 * @param [in]: Command line parameter
 * @config [in]: Application configuration structure
 * @return: DOCA_SUCCESS on success and DOCA_ERROR_INVALID_VALUE otherwise
 */
static doca_error_t
queue_num_callback(void *param, void *config)
{
	struct sample_send_wait_cfg *sample_cfg = (struct sample_send_wait_cfg *)config;
	int queue_num = *((int *)param);

	if (queue_num == 0 || queue_num > MAX_QUEUES) {
		DOCA_LOG_ERR("GPU receive queue number is wrong 0 < %d < %d", queue_num, MAX_QUEUES);
		return DOCA_ERROR_INVALID_VALUE;
	}

	sample_cfg->queue_num = queue_num;

	return DOCA_SUCCESS;
}

/*
 * Get GPU receive queue number.
 *
 * @param [in]: Command line parameter
 * @config [in]: Application configuration structure
 * @return: DOCA_SUCCESS on success and DOCA_ERROR_INVALID_VALUE otherwise
 */
static doca_error_t
workload_callback(void *param, void *config)
{
	struct sample_send_wait_cfg *sample_cfg = (struct sample_send_wait_cfg *)config;
	int workload = *((int *)param);

	if (workload < 0 || workload > MAX_WORKLOAD) {
		DOCA_LOG_ERR("Workload number is wrong 0 < %d < %d", workload, MAX_WORKLOAD);
		return DOCA_ERROR_INVALID_VALUE;
	}

	sample_cfg->workload = workload;

	return DOCA_SUCCESS;
}

/*
 * Get GPU receive queue number.
 *
 * @param [in]: Command line parameter
 * @config [in]: Application configuration structure
 * @return: DOCA_SUCCESS on success and DOCA_ERROR_INVALID_VALUE otherwise
 */
static doca_error_t
batch_callback(void *param, void *config)
{
	struct sample_send_wait_cfg *sample_cfg = (struct sample_send_wait_cfg *)config;
	int batch = *((int *)param);

	if (batch < 0 || batch > MAX_RX_NUM_PKTS) {
		DOCA_LOG_ERR("Batch number is wrong 0 < %d < %d", batch, MAX_RX_NUM_PKTS);
		return DOCA_ERROR_INVALID_VALUE;
	}

	sample_cfg->batch = batch;

	return DOCA_SUCCESS;
}

/*
 * Register sample command line parameters.
 *
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
register_sample_params(void)
{
	doca_error_t result;
	struct doca_argp_param *gpu_param, *nic_param, *queue_param, *workload_param, *batch_param;

	result = doca_argp_param_create(&gpu_param);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create ARGP param: %s", doca_error_get_descr(result));
		return result;
	}

	doca_argp_param_set_short_name(gpu_param, "g");
	doca_argp_param_set_long_name(gpu_param, "gpu");
	doca_argp_param_set_arguments(gpu_param, "<GPU PCIe address>");
	doca_argp_param_set_description(gpu_param, "GPU PCIe address to be used by the sample");
	doca_argp_param_set_callback(gpu_param, gpu_pci_address_callback);
	doca_argp_param_set_type(gpu_param, DOCA_ARGP_TYPE_STRING);
	doca_argp_param_set_mandatory(gpu_param);
	result = doca_argp_register_param(gpu_param);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to register ARGP param: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_argp_param_create(&nic_param);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create ARGP param: %s", doca_error_get_descr(result));
		return result;
	}

	doca_argp_param_set_short_name(nic_param, "n");
	doca_argp_param_set_long_name(nic_param, "nic");
	doca_argp_param_set_arguments(nic_param, "<NIC PCIe address>");
	doca_argp_param_set_description(nic_param, "DOCA device PCIe address used by the sample");
	doca_argp_param_set_callback(nic_param, nic_pci_address_callback);
	doca_argp_param_set_type(nic_param, DOCA_ARGP_TYPE_STRING);
	doca_argp_param_set_mandatory(nic_param);
	result = doca_argp_register_param(nic_param);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to register ARGP param: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_argp_param_create(&queue_param);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create ARGP param: %s", doca_error_get_descr(result));
		return result;
	}

	doca_argp_param_set_short_name(queue_param, "q");
	doca_argp_param_set_long_name(queue_param, "queue");
	doca_argp_param_set_arguments(queue_param, "<GPU receive queues>");
	doca_argp_param_set_description(queue_param, "DOCA GPUNetIO receive queue per flow");
	doca_argp_param_set_callback(queue_param, queue_num_callback);
	doca_argp_param_set_type(queue_param, DOCA_ARGP_TYPE_INT);
	doca_argp_param_set_mandatory(queue_param);
	result = doca_argp_register_param(queue_param);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to register ARGP param: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_argp_param_create(&workload_param);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create ARGP param: %s", doca_error_get_descr(result));
		return result;
	}

	doca_argp_param_set_short_name(workload_param, "w");
	doca_argp_param_set_long_name(workload_param, "workload");
	doca_argp_param_set_arguments(workload_param, "<Type of wl>");
	doca_argp_param_set_description(workload_param, "EtherMirror, IPLookup or CRC");
	doca_argp_param_set_callback(workload_param, workload_callback);
	doca_argp_param_set_type(workload_param, DOCA_ARGP_TYPE_INT);
	doca_argp_param_set_mandatory(workload_param);
	result = doca_argp_register_param(workload_param);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to register ARGP param: %s", doca_error_get_descr(result));
		return result;
	}

	result = doca_argp_param_create(&batch_param);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create ARGP param: %s", doca_error_get_descr(result));
		return result;
	}

	doca_argp_param_set_short_name(batch_param, "b");
	doca_argp_param_set_long_name(batch_param, "batch");
	doca_argp_param_set_arguments(batch_param, "<Size of batch>");
	doca_argp_param_set_description(batch_param, "Size of batch");
	doca_argp_param_set_callback(batch_param, batch_callback);
	doca_argp_param_set_type(batch_param, DOCA_ARGP_TYPE_INT);
	doca_argp_param_set_mandatory(batch_param);
	result = doca_argp_register_param(batch_param);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to register ARGP param: %s", doca_error_get_descr(result));
		return result;
	}

	return DOCA_SUCCESS;
}

/*
 * Sample main function
 *
 * @argc [in]: command line arguments size
 * @argv [in]: array of command line arguments
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
int
main(int argc, char **argv)
{
	doca_error_t result;
	struct doca_log_backend *sdk_log;
	struct sample_send_wait_cfg sample_cfg;
	int exit_status = EXIT_FAILURE;
	int cuda_id;
	cudaError_t cuda_ret;

	/* Register a logger backend */
	result = doca_log_backend_create_standard();
	if (result != DOCA_SUCCESS)
		goto sample_exit;

	/* Register a logger backend for internal SDK errors and warnings */
	result = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
	if (result != DOCA_SUCCESS)
		goto sample_exit;
	result = doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_WARNING);
	if (result != DOCA_SUCCESS)
		goto sample_exit;

	DOCA_LOG_INFO("Starting the sample");

	result = doca_argp_init("doca_gpunetio_simple_receive", &sample_cfg);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init ARGP resources: %s", doca_error_get_descr(result));
		goto sample_exit;
	}

	result = register_sample_params();
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to parse application input: %s", doca_error_get_descr(result));
		goto argp_cleanup;
	}

	result = doca_argp_start(argc, argv);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to parse application input: %s", doca_error_get_descr(result));
		goto argp_cleanup;
	}

	printf("wl nbr: %d\n", sample_cfg.workload);

	/* In a multi-GPU system, ensure CUDA refers to the right GPU device */
	cuda_ret = cudaDeviceGetByPCIBusId(&cuda_id, sample_cfg.gpu_pcie_addr);
	if (cuda_ret != cudaSuccess) {
		DOCA_LOG_ERR("Invalid GPU bus id provided %s", sample_cfg.gpu_pcie_addr);
		return DOCA_ERROR_INVALID_VALUE;
	}

	cudaFree(0);
	cudaSetDevice(cuda_id);

	DOCA_LOG_INFO("Sample configuration:\n\tGPU\t%s\n\tNIC\t%s\n\tQueues\t%d\n\t",
		sample_cfg.gpu_pcie_addr, sample_cfg.nic_pcie_addr, sample_cfg.queue_num);

	result = gpunetio_simple_receive(&sample_cfg);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("gpunetio_simple_receive() encountered an error: %s", doca_error_get_descr(result));
		goto argp_cleanup;
	}

	exit_status = EXIT_SUCCESS;

argp_cleanup:
	doca_argp_destroy();
sample_exit:
	if (exit_status == EXIT_SUCCESS)
		DOCA_LOG_INFO("Sample finished successfully");
	else
		DOCA_LOG_INFO("Sample finished with errors");
	return exit_status;
}
