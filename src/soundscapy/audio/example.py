import multiprocessing

from soundscapy.audio.processing_engine import ProcessingEngine, PsychoacousticProcessor
from soundscapy.audio.state_manager import StateManager
from soundscapy.logging import set_log_level


def main(
    single_file_path="/Users/mitch/Documents/GitHub/Soundscapy/test/data/CT101.wav",
    directory_path="/Users/mitch/Documents/GitHub/Soundscapy/test/data",
):
    set_log_level("DEBUG")

    import os
    from soundscapy.audio.result_storage import (
        FileAnalysisResults,
        DirectoryAnalysisResults,
    )

    # Configure the PsychoacousticProcessor
    config = {"metrics": {"loudness_zwtv": {"enabled": True}}}

    processor = PsychoacousticProcessor(config)
    processing_engine = ProcessingEngine(processor, max_workers=10)
    state_file_path = "test_state.json"
    state_manager = StateManager(state_file_path)

    # Test processing a single file
    # single_file_path = "/Users/mitch/Documents/GitHub/Soundscapy/test/data/CT101.wav"

    single_file_result = processing_engine._process_file(
        single_file_path, processor, state_manager, parallel_channels=True
    )
    print(f"Single file result: {single_file_result}")

    # Check state after processing
    assert state_manager.is_processed(
        single_file_path
    ), "File should be marked as processed after processing"

    # Save single file result
    single_file_result_path = "single_file_result.h5"
    single_file_result.save(single_file_result_path)
    print(f"Saved single file result to: {single_file_result_path}")

    # Load single file result
    loaded_single_file_result = FileAnalysisResults(single_file_path)
    loaded_single_file_result.load(single_file_result_path)
    print(f"Loaded single file result: {loaded_single_file_result}")

    # Compare original and loaded results
    assert (
        single_file_result.file_path == loaded_single_file_result.file_path
    ), "File paths do not match"
    assert (
        single_file_result.metrics.keys() == loaded_single_file_result.metrics.keys()
    ), "Metrics do not match"
    print("Single file result save and load successful")

    # Test processing a directory
    # directory_path = "/Users/mitch/Documents/GitHub/Soundscapy/test/data"

    # Get list of audio files in the directory
    audio_files = [f for f in os.listdir(directory_path) if f.endswith(".wav")]

    directory_results = processing_engine.process_directory(
        directory_path, state_manager
    )
    print(f"Directory results: {directory_results}")

    # Check state after processing the directory
    for audio_file in audio_files:
        file_path = os.path.join(directory_path, audio_file)
        assert state_manager.is_processed(
            file_path
        ), f"File {file_path} should be marked as processed after directory processing"

    # Save directory results
    directory_results_path = "directory_results.h5"
    directory_results.save(directory_results_path)
    print(f"Saved directory results to: {directory_results_path}")

    # Load directory results
    loaded_directory_results = DirectoryAnalysisResults(directory_path)
    loaded_directory_results.load(directory_results_path)
    print(f"Loaded directory results: {loaded_directory_results}")

    # Compare original and loaded directory results
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

    print("Directory results save and load successful")

    # Test state persistence
    new_state_manager = StateManager(state_file_path)
    for audio_file in audio_files:
        file_path = os.path.join(directory_path, audio_file)
        assert new_state_manager.is_processed(
            file_path
        ), f"File {file_path} should still be marked as processed after reloading state"

    print("State manager persistence test successful")

    # Test force reprocessing
    force_reprocess_config = config.copy()
    force_reprocess_config["force_reprocess"] = True
    force_reprocess_processor = PsychoacousticProcessor(force_reprocess_config)

    force_reprocess_result = processing_engine._process_file(
        single_file_path, force_reprocess_processor, new_state_manager
    )
    print(f"Force reprocess result: {force_reprocess_result}")

    # Clean up test files
    # os.remove(single_file_result_path)
    # os.remove(directory_results_path)
    os.remove(state_file_path)
    print("Test files cleaned up")

    # Test error handling
    non_existent_file = "non_existent_file.wav"
    error_result = processing_engine._process_file(
        non_existent_file, processor, state_manager
    )
    print(f"Error result: {error_result}")
    assert error_result.has_errors(), "Error result should have errors"
    assert (
        len(error_result.errors) > 0
    ), "Error result should have at least one error message"
    print(f"Error messages: {error_result.errors}")

    # Save and load error result
    error_result_path = "error_result.h5"
    error_result.save(error_result_path)
    loaded_error_result = FileAnalysisResults(non_existent_file)
    loaded_error_result.load(error_result_path)
    assert loaded_error_result.has_errors(), "Loaded error result should have errors"
    assert (
        loaded_error_result.errors == error_result.errors
    ), "Loaded error messages should match original"
    print("Error result save and load successful")

    # Clean up error result file
    os.remove(error_result_path)

    print("All tests completed successfully")


def benchmark(executor_class, max_workers, directory_path, state_manager):
    start_time = time.time()

    with executor_class(max_workers=max_workers) as executor:
        # Your existing process_directory logic here
        pass

    end_time = time.time()
    return end_time - start_time


if __name__ == "__main__":
    import time
    from loguru import logger

    multiprocessing.freeze_support()  # This line is needed if you're using PyInstaller or similar tools
    single_file_path = "/Users/mitch/Documents/GitHub/Soundscapy/test/data/CT101.wav"
    directory_path = "/Users/mitch/Documents/GitHub/Soundscapy/test/data"

    start = time.time()
    main(single_file_path, directory_path)

    logger.info(f"Time taken: {time.time() - start} seconds")
