from pathlib import Path

# Constants and Labels
PARAM_LIST = [
    "LevelA",
    "LevelC",
    "LevelZ",
    "Loudness",
    "Roughness",
    "Sharpness",
    "Tonality",
    "FluctuationStrength",
    "SIL",
    "THD",
    "Impulsiveness",
]

LOCATION_IDS = {
    "London": [
        "CamdenTown",
        "EustonTap",
        "MarchmontGarden",
        "PancrasLock",
        "RegentsParkFields",
        "RegentsParkJapan",
        "RussellSq",
        "StPaulsCross",
        "StPaulsRow",
        "TateModern",
        "TorringtonSq",
    ],
    "Venice": ["SanMarco", "MonumentoGaribaldi",],
    "Granada": ["CampoPrincipe", "CarloV", "MiradorSanNicolas", "PlazaBibRambla",],
    "Groningen": ["GroningenNoorderplantsoen"],
    "Test": ["LocationA", "LocationB"],
}

IGNORE_LIST = ["AllLondon", "AllyPally", "CoventGd1", "OxfordSt"]


def collect_param_dirs(
    root_directory: Path,
    location_ids: list,
    param_list: list,
    include_TS: bool = True,
    include_spectrum: bool = True,
):
    """Iterate throughout the SSID DB file structure to extract TimeSeries, SpectrumData fil paths.

    Parameters
    ----------
    root_directory : Path
        The city-level SSID directory
    location_ids : list
        A subset of LocationIDs to include in the filepath collection
    param_list : list
        A subset of parameters to include in the filepath collection
    include_TS : bool, optional
        Collect TimeSeries files?, by default True
    include_spectrum : bool, optional
        Collect SpectrumData files?, by default True

    Returns
    -------
    tuple of lists
        A tuple containing the full lists of TimeSeries and SpectrumData files.
        These lists contain WindowsPath objects.
    
    Examples
    ________
    >>> full_ts_list, full_spectrum_list = collect_param_dirs(TEST_DIR, LOCATION_IDS["Test"], PARAM_LIST)
    >>> len(full_ts_list)
    33
    >>> len(full_spectrum_list)
    6
    """

    # Collect list of session id directories
    session_dirs = [session for session in root_directory.iterdir() if session.is_dir()]
    session_dirs = [session for session in session_dirs if "OFF" in session.name]

    new_session_dirs = [
        [session for location in location_ids if (location in session.name)]
        for session in session_dirs
    ]

    session_dirs = [
        val for sublist in new_session_dirs for val in sublist
    ]  # Remove blank entries

    bin_dirs = []
    for session in session_dirs:
        bin_dir = [
            child
            for child in session.iterdir()
            if child.is_dir() and "BIN" in child.name
        ][0]
        bin_dirs.append(bin_dir)

    if include_TS:
        # TODO: separate into own function
        # Collect Time Series parameter directories
        ts_dirs = []
        for directory in bin_dirs:
            ts_dir = [
                child
                for child in directory.iterdir()
                if child.is_dir() and "TimeSeries" in child.name
            ][0]
            ts_dirs.append(ts_dir)

        param_dirs = []
        for directory in ts_dirs:
            param_dir = [child for child in directory.iterdir() if child.is_dir()]
            param_dirs.append(param_dir)
        param_dirs = [val for sublist in param_dirs for val in sublist]

        full_ts_list = [
            [directory for param in param_list if (param + "_TS" in directory.name)]
            for directory in param_dirs
        ]

        full_ts_list = [val for sublist in full_ts_list for val in sublist]

    # Collect Spectrum directories
    # TODO: separate into own function

    spectrum_dirs = []
    if include_spectrum:
        for directory in bin_dirs:
            spectrum_dir = [
                child
                for child in directory.iterdir()
                if child.is_dir() and "SpectrumData" in child.name
            ][0]
            spectrum_dirs.append(spectrum_dir)

        full_spectrum_list = []
        for directory in spectrum_dirs:
            spectrum_dir = [child for child in directory.iterdir() if child.is_dir()]
            full_spectrum_list.append(spectrum_dir)

        full_spectrum_list = [val for sublist in full_spectrum_list for val in sublist]

    # TODO: Add include_wav

    return full_ts_list, full_spectrum_list


if __name__ == "__main__":
    import doctest

    TEST_DIR = Path("test_DB")
    doctest.testmod(verbose=False, optionflags=doctest.ELLIPSIS)
