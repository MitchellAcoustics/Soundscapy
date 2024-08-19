import os

import numpy as np
import pytest
from loguru import logger
from soundscapy.audio.processing_engine import ProcessingEngine, PsychoacousticProcessor
from soundscapy.audio.result_storage import (
    DirectoryAnalysisResults,
    FileAnalysisResults,
)
from soundscapy.audio.state_manager import StateManager
from soundscapy.logging import set_log_level


@pytest.fixture(scope="module")
def setup_test_environment(request):
    set_log_level("DEBUG")
    config = {"metrics": {"loudness_zwtv": {"enabled": True}}}
    processor = PsychoacousticProcessor(config)
    processing_engine = ProcessingEngine(processor)
    state_file_path = "test_state.json"
    state_manager = StateManager(state_file_path)

    def finalizer():
        logger.info("Cleaning up test environment")
        if os.path.exists(state_file_path):
            os.remove(state_file_path)

    request.addfinalizer(finalizer)

    return processor, processing_engine, state_manager, state_file_path


@pytest.fixture(scope="module")
def test_file_paths():
    single_file_path = "/Users/mitch/Documents/GitHub/Soundscapy/test/data/CT101.wav"
    directory_path = "/Users/mitch/Documents/GitHub/Soundscapy/test/data"
    return single_file_path, directory_path


@pytest.fixture
def cleanup_files(request):
    files_to_cleanup = []

    def add_file(file_path):
        files_to_cleanup.append(file_path)

    def finalizer():
        for file_path in files_to_cleanup:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up file: {file_path}")

    request.addfinalizer(finalizer)

    return add_file


@pytest.mark.slow
def test_save_and_load_single_file_result(
    setup_test_environment, test_file_paths, cleanup_files
):
    processor, processing_engine, state_manager, _ = setup_test_environment
    single_file_path, _ = test_file_paths

    assert not state_manager.is_processed(
        single_file_path
    ), "File should not be marked as processed initially"

    single_file_result = processing_engine._process_file(
        single_file_path, processor, state_manager
    )
    logger.debug(f"Single file result: {single_file_result}")

    assert state_manager.is_processed(
        single_file_path
    ), "File should be marked as processed after processing"

    single_file_result_path = "single_file_result.h5"
    cleanup_files(single_file_result_path)

    single_file_result.save(single_file_result_path)
    logger.info(f"Saved single file result to: {single_file_result_path}")

    loaded_single_file_result = FileAnalysisResults(single_file_path)
    loaded_single_file_result.load(single_file_result_path)
    logger.debug(f"Loaded single file result: {loaded_single_file_result}")

    assert (
        single_file_result.file_path == loaded_single_file_result.file_path
    ), "File paths do not match"
    assert (
        single_file_result.metrics.keys() == loaded_single_file_result.metrics.keys()
    ), "Metrics do not match"

    for metric_name, original_result in single_file_result.metrics.items():
        loaded_result = loaded_single_file_result.metrics[metric_name]
        logger.debug(f"Original {metric_name} result: {original_result}")
        logger.debug(f"Loaded {metric_name} result: {loaded_result}")

        for channel, original_channel_result in original_result.channels.items():
            loaded_channel_result = loaded_result.channels[channel]
            logger.debug(
                f"Original {metric_name} {channel} result: {original_channel_result}"
            )
            logger.debug(
                f"Loaded {metric_name} {channel} result: {loaded_channel_result}"
            )

            assert np.allclose(
                original_channel_result.N, loaded_channel_result.N
            ), f"N values do not match for {metric_name} {channel}"
            assert np.allclose(
                original_channel_result.N_specific, loaded_channel_result.N_specific
            ), f"N_specific values do not match for {metric_name} {channel}"

    logger.info("Single file result save and load successful")

    return single_file_result


# def test_save_and_load_single_file_result(
#     setup_test_environment, test_file_paths, cleanup_files
# ):
#     single_file_result = test_single_file_processing(
#         setup_test_environment, test_file_paths
#     )
#     single_file_path, _ = test_file_paths
#
#     single_file_result_path = "single_file_result.h5"
#     cleanup_files(single_file_result_path)
#
#     single_file_result.save(single_file_result_path)
#     logger.info(f"Saved single file result to: {single_file_result_path}")
#
#     loaded_single_file_result = FileAnalysisResults(single_file_path)
#     loaded_single_file_result.load(single_file_result_path)
#     logger.debug(f"Loaded single file result: {loaded_single_file_result}")
#
#     assert (
#         single_file_result.file_path == loaded_single_file_result.file_path
#     ), "File paths do not match"
#     assert (
#         single_file_result.metrics.keys() == loaded_single_file_result.metrics.keys()
#     ), "Metrics do not match"
#
#     for metric_name, original_result in single_file_result.metrics.items():
#         loaded_result = loaded_single_file_result.metrics[metric_name]
#         logger.debug(f"Original {metric_name} result: {original_result}")
#         logger.debug(f"Loaded {metric_name} result: {loaded_result}")
#
#         for channel, original_channel_result in original_result.channels.items():
#             loaded_channel_result = loaded_result.channels[channel]
#             logger.debug(
#                 f"Original {metric_name} {channel} result: {original_channel_result}"
#             )
#             logger.debug(
#                 f"Loaded {metric_name} {channel} result: {loaded_channel_result}"
#             )
#
#             assert np.allclose(
#                 original_channel_result.N, loaded_channel_result.N
#             ), f"N values do not match for {metric_name} {channel}"
#             assert np.allclose(
#                 original_channel_result.N_specific, loaded_channel_result.N_specific
#             ), f"N_specific values do not match for {metric_name} {channel}"
#
#     logger.info("Single file result save and load successful")


def test_directory_processing(setup_test_environment, test_file_paths):
    processor, processing_engine, state_manager, _ = setup_test_environment
    _, directory_path = test_file_paths

    directory_results = processing_engine.process_directory(
        directory_path, state_manager
    )
    logger.debug(f"Directory results: {directory_results}")

    return directory_results


def test_save_and_load_directory_results(
    setup_test_environment, test_file_paths, cleanup_files
):
    directory_results = test_directory_processing(
        setup_test_environment, test_file_paths
    )
    _, directory_path = test_file_paths

    directory_results_path = "directory_results.h5"
    cleanup_files(directory_results_path)

    directory_results.save(directory_results_path)
    logger.info(f"Saved directory results to: {directory_results_path}")

    loaded_directory_results = DirectoryAnalysisResults(directory_path)
    loaded_directory_results.load(directory_results_path)
    logger.debug(f"Loaded directory results: {loaded_directory_results}")

    assert (
        directory_results.directory_path == loaded_directory_results.directory_path
    ), "Directory paths do not match"
    assert set(directory_results.file_results.keys()) == set(
        loaded_directory_results.file_results.keys()
    ), "Processed files do not match"

    for file_path in directory_results.file_results.keys():
        original_file_result = directory_results.file_results[file_path]
        loaded_file_result = loaded_directory_results.file_results[file_path]
        assert (
            original_file_result.metrics.keys() == loaded_file_result.metrics.keys()
        ), f"Metrics do not match for {file_path}"

    logger.info("Directory results save and load successful")


def test_error_handling(setup_test_environment, cleanup_files):
    processor, processing_engine, state_manager, _ = setup_test_environment

    non_existent_file = "non_existent_file.wav"
    error_result = processing_engine._process_file(
        non_existent_file, processor, state_manager
    )
    logger.debug(f"Error result: {error_result}")

    assert error_result.has_errors(), "Error result should have errors"
    assert (
        len(error_result.errors) > 0
    ), "Error result should have at least one error message"
    logger.debug(f"Error messages: {error_result.errors}")

    error_result_path = "error_result.h5"
    cleanup_files(error_result_path)

    error_result.save(error_result_path)
    loaded_error_result = FileAnalysisResults(non_existent_file)
    loaded_error_result.load(error_result_path)

    assert loaded_error_result.has_errors(), "Loaded error result should have errors"
    assert (
        loaded_error_result.errors == error_result.errors
    ), "Loaded error messages should match original"
    logger.info("Error result save and load successful")
