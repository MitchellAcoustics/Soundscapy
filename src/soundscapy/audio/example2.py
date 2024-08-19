import os

from soundscapy.analyzer import create_analyzer

from loguru import logger


def main():
    # Create an analyzer with default configuration
    analyzer = create_analyzer()

    # Analyze a single file
    single_file_path = "path/to/audio/file.wav"
    single_file_result = analyzer["analyze_file"](single_file_path)
    logger.debug(f"Single file result: {single_file_result}")

    # Save single file result
    single_file_result_path = "single_file_result.h5"
    single_file_result.save(single_file_result_path)
    logger.info(f"Saved single file result to: {single_file_result_path}")

    # Load single file result
    loaded_single_file_result = analyzer["create_file_analysis_results"](
        single_file_path
    )
    loaded_single_file_result.load(single_file_result_path)
    logger.debug(f"Loaded single file result: {loaded_single_file_result}")

    # Compare original and loaded results
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

    # Clean up test files
    os.remove(single_file_result_path)
    logger.info("Test files cleaned up")

    logger.info("All tests completed successfully")


if __name__ == "__main__":
    main()
