import multiprocessing

from soundscapy.audio.processing_engine import ProcessingEngine, PsychoacousticProcessor
from soundscapy.audio.state_manager import StateManager
from soundscapy.logging import set_log_level


def main():
    set_log_level("DEBUG")

    # Configure the PsychoacousticProcessor
    config = {"metrics": {"loudness_zwtv": {"enabled": True}}}

    processor = PsychoacousticProcessor(config)
    processing_engine = ProcessingEngine(processor, max_workers=10)
    state_manager = StateManager("processing_state.json")

    # Process a directory
    directory_path = "/Users/mitch/Documents/GitHub/Soundscapy/test/data"
    directory_results = processing_engine.process_directory(
        directory_path, state_manager
    )

    # Get combined stats
    combined_stats = processing_engine.get_combined_stats(directory_results)

    # Save the results
    directory_results.save("directory_results.h5")

    print("Processing completed successfully!")
    print(combined_stats)


if __name__ == "__main__":
    multiprocessing.freeze_support()  # This line is needed if you're using PyInstaller or similar tools
    main()
