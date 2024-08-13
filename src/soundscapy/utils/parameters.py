# Constants and Labels
PAQ_NAMES = [
    "pleasant",
    "vibrant",
    "eventful",
    "chaotic",
    "annoying",
    "monotonous",
    "uneventful",
    "calm",
]

PAQ_IDS = [
    "PAQ1",
    "PAQ2",
    "PAQ3",
    "PAQ4",
    "PAQ5",
    "PAQ6",
    "PAQ7",
    "PAQ8",
]

IGNORE_LIST = ["AllLondon", "AllyPally", "CoventGd1", "OxfordSt"]

CATEGORISED_VARS = {
    "indexing": [
        "GroupID",
        "SessionID",
        "LocationID",
        "Country",
        "record_id",
    ],  # Ways to index which survey it is
    "meta_info": [
        "recording",
        "start_time",
        "end_time",
        "longitude",
        "latitude",
    ],  # Info from when that survey was collected
    "sound_source_dominance": [
        "Traffic",
        "Other",
        "Human",
        "Natural",
    ],  # Sound sources
    "complex_PAQs": ["Pleasant", "Eventful"],  # Projected PAQ coordinates
    "raw_PAQs": [
        "pleasant",
        "chaotic",
        "vibrant",
        "uneventful",
        "calm",
        "annoying",
        "eventful",
        "monotonous",
    ],  # Raw 5-point PAQs
    "overall_soundscape": [
        "overall",
        "appropriateness",
        "loudness",
        "often",
        "visit_again",
    ],  # Questions about the overall soundscape
    "demographics": ["Age", "Gender", "Occupation", "Education", "Ethnicity", "Resid"],
    "misc": ["AnythingElse"],
}

EQUAL_ANGLES = (0, 45, 90, 135, 180, 225, 270, 315)

# Adjusted angles as defined in Aletta et. al. (2024)
LANGUAGE_ANGLES = {
    "eng": (0, 46, 94, 138, 177, 241, 275, 340),
    "arb": (0, 36, 45, 135, 167, 201, 242, 308),
    "cmn": (0, 18, 38, 154, 171, 196, 217, 318),
    "hrv": (0, 84, 93, 160, 173, 243, 273, 354),
    "nld": (0, 43, 111, 125, 174, 257, 307, 341),
    "deu": (0, 64, 97, 132, 182, 254, 282, 336),
    "ell": (0, 72, 86, 133, 161, 233, 267, 328),
    "ind": (0, 53, 104, 123, 139, 202, 284, 308),
    "ita": (0, 57, 104, 143, 170, 274, 285, 336),
    "spa": (0, 41, 103, 147, 174, 238, 279, 332),
    "swe": (0, 66, 87, 146, 175, 249, 275, 335),
    "tur": (0, 55, 97, 106, 157, 254, 289, 313),
}
