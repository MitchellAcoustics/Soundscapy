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

ssid_v1_lvls = {
    "sound_source_dominance": {
        "type": "Likert",
        "levels": {
            1: "Not at all",
            2: "A little",
            3: "Moderately",
            4: "A lot",
            5: "Dominates completely",
        },
    },
    "PAQ": {
        "type": "Likert",
        "levels": {
            1: "Strongly disagree",
            2: "Somewhat disagree",
            3: "Neither agree nor disagree",
            4: "Somewhat agree",
            5: "Strongly agree",
        },
    },
    "overall_soundscape": {
        "type": "Likert",
        "levels": {
            1: "Strongly disagree",
            2: "Somewhat disagree",
            3: "Neither agree nor disagree",
            4: "Somewhat agree",
            5: "Strongly agree",
        },
    },
    "who": {
        "type": "ordinal",
        "levels": {
            0: "At no time",
            1: "Some of the time",
            2: "Less than half of the time",
            3: "More than half of the time",
            4: "Most of the time",
            5: "All of the time",
        },
    },
}

ssid_v1_aliases = {
    "GroupID": ["group_id"],
    "SessionID": ["session_id"],
    "LocationID": ["location_id"],
    "RecordID": ["record_id"],
    "Traffic": ["ssi01"],
    "Other": ["ssi02"],
    "Human": ["ssi03"],
    "Natural": ["ssi04"],
    "pleasant": ["paq01", "pl"],
    "chaotic": ["paq02", "ch"],
    "vibrant": ["paq03", "vi", "vib"],
    "uneventful": ["paq04", "un"],
    "calm": ["paq05", "ca"],
    "annoying": ["paq06", "an"],
    "monotonous": ["paq08", "mo", "mono"],
    "overall": ["sss01"],
    "appropriateness": ["sss02", "appropriate", "approp"],
    "loudness": ["sss03"],
    "often": ["sss04"],
    "visit_again": ["sss05"],
    "who01": ["who1"],
    "who02": ["who2"],
    "who03": ["who3"],
    "who04": ["who4"],
    "who05": ["who5"],
    "WHO_sum": ["who"],
    "Age": ["age00"],
    "Gender": ["gen", "gen00"],
    "Occupation": ["occ00", "occup", "occ"],
    "Education": ["edu00", "edu"],
    "Ethnicity": ["eth00", "eth"],
    "Resid": ["misc00", "Resident"],
    "AnythingElse": ["misc01"],
}


class SurveyQ:
    def __init__(
        self,
        label,
        category,
        version="ssid_v1",
        aliases=None,
        type=None,
        levels=None,
        text="",
    ):
        self.label = label
        self.category = category
        self.text = text

        if category in ssid_v1_lvls.keys() and levels is None:
            self.levels = ssid_v1_lvls[category]["levels"]
            self.type = ssid_v1_lvls[category]["type"]
        else:
            self.levels = levels
            if type is None:
                raise AttributeError(
                    "Need to define a type if self-defining the levels"
                )
            else:
                self.type = type

        if label in ssid_v1_aliases.keys() and aliases is None:
            self.aliases = ssid_v1_aliases[label]
        else:
            self.aliases = aliases


# ? Add a 'type' tag? i.e. "type": "Likert" or "type": "Ordinal"
SURVEY_VARS = {
    "GroupID": {"aliases": ["group_id"],},
    "SessionID": {"aliases": ["session_id"],},
    "LocationID": {"aliases": ["location_id"],},
    "RecordID": {"aliases": ["record_id"],},
    "Traffic": {
        "aliases": ["ssi01"],
        "levels": ssid_v1_lvls["sound_source_dominance"],
        "text": "Traffic noise (e.g. cars, buses, trains, airplanes)",
    },
    "Other": {
        "aliases": ["ssi02"],
        "levels": ssid_v1_lvls["sound_source_dominance"],
        "text": "Other noise (e.g. sirens, construction, industry, loadining of goods)",
    },
    "Human": {
        "aliases": ["ssi03"],
        "levels": ssid_v1_lvls["sound_source_dominance"],
        "text": "Sounds from human beings (e.g. conversation, laughter, children at play, footsteps)",
    },
    "Natural": {
        "aliases": ["ssi04"],
        "levels": ssid_v1_lvls["sound_source_dominance"],
        "text": "Natural sounds (e.g. singing birds, flowing water, wind in vegetation)",
    },
    "pleasant": {
        "aliases": ["paq01", "pl"],
        "levels": ssid_v1_lvls["PAQ"],
        "text": "Pleasant",
    },
    "chaotic": {
        "aliases": ["paq02", "ch"],
        "levels": ssid_v1_lvls["PAQ"],
        "text": "Chaotic",
    },
    "vibrant": {
        "aliases": ["paq03", "vi", "vib"],
        "levels": ssid_v1_lvls["PAQ"],
        "text": "Vibrant",
    },
    "uneventful": {
        "aliases": ["paq04", "un"],
        "levels": ssid_v1_lvls["PAQ"],
        "text": "Uneventful",
    },
    "calm": {"aliases": ["paq05", "ca"], "levels": ssid_v1_lvls["PAQ"],},
    "annoying": {"aliases": ["paq06", "an"], "levels": ssid_v1_lvls["PAQ"],},
    "eventful": {
        "aliases": ["paq07", "ev"],
        "levels": ssid_v1_lvls["PAQ"],
        "text": "Eventful",
    },
    "monotonous": {
        "aliases": ["paq08", "mo", "mono"],
        "levels": ssid_v1_lvls["PAQ"],
        "text": "Monotonous",
    },
    "overall": {
        "aliases": ["sss01"],
        "levels": ssid_v1_lvls["overall_soundscape"],
        "text": "Overall, how would you descrive the present surrounding sound environment?",
    },
    "appropriateness": {
        "aliases": ["sss02", "appropriate", "approp"],
        "levels": ssid_v1_lvls["overall_soundscape"],
        "text": "Overall, to what extent is the present surrounding sound environment approrpriate to the present place?",
    },
    "loudness": {
        "aliases": ["sss03"],
        "levels": ssid_v1_lvls["overall_soundscape"],
        "text": "How loud would you say the sound environment is?",
    },
    "often": {
        "aliases": ["sss04"],
        "levels": ssid_v1_lvls["overall_soundscape"],
        "text": "How often do you visit this place?",
    },
    "visit_again": {
        "aliases": ["sss05"],
        "levels": ssid_v1_lvls["overall_soundscape"],
        "text": "How often would you like to visit this place again?",
    },
    "who01": {
        "aliases": ["WHO01"],
        "levels": ssid_v1_lvls["who"],
        "text": "I have felt cheerful and in good spirits.",
    },
    "who02": {
        "aliases": ["WHO02"],
        "levels": ssid_v1_lvls["who"],
        "text": "I have felt calm and relaxed.",
    },
    "who03": {
        "aliases": ["WHO03"],
        "levels": ssid_v1_lvls["who"],
        "text": "I have felt active and vigorous.",
    },
    "who04": {
        "aliases": ["WHO04"],
        "levels": ssid_v1_lvls["who"],
        "text": "I woke up feeling fresh and rested.",
    },
    "who05": {
        "aliases": ["WHO05"],
        "levels": ssid_v1_lvls["who"],
        "text": "My daily life has been filled with things that interest me.",
    },
    "WHO_sum": {"aliases": ["WHO_SUM", "who"], "range": (1, 100),},
    "Age": {
        "aliases": ["age", "age00"],
        "range": (18, 100),
        "text": "How old are you?",
    },
    "Gender": {
        "aliases": ["gen", "gen00"],
        "levels": {1: "Male", 2: "Female", 3: "Non-conforming", 4: "Rather not say"},
        "text": "What is your gender?",
    },
    "Occupation": {
        "aliases": ["occ00", "occup", "occ"],
        "levels": {
            1: "Employed",
            2: "Unemployed",
            3: "Retired",
            4: "Student",
            5: "Other",
            6: "Rather not say",
        },
        "text": "What is your occupational status?",
    },
    "Education": {
        "aliases": ["edu00", "edu"],
        "levels": {
            1: "Some high school",
            2: "High school graduate",
            3: "Some college",
            4: "Trade-Technical-Vocational training",
            5: "University graduate",
            6: "Some postgraduate work",
            7: "Postgraduate degree",
        },
        "text": "What is the highest level of education you have completed?",
    },
    "Ethnicity": {
        "aliases": ["eth00", "eth"],
        "levels": {
            1: "White",
            2: "Mixed-multiple ethnic groups",
            3: "Asian-Asian British",
            4: "Black-African-Caribbean-Black British",
            5: "Middle Eastern",
            6: "Rather not say",
            7: "Other ethnic group",
        },
    },
    "Resid": {
        "aliases": ["misc00", "Resident"],
        "levels": {1: "A local", 2: "A tourist", 3: "Other",},
        "text": "Would you consider yourself...",
    },
    "AnythingElse": {
        "aliases": ["misc01"],
        "levels": None,
        "text": "Is there anything else you want to let us know about the sound environment?",
    },
}

