#!/usr/bin/env python3
"""
generate_dataset.py
Deterministic generator for a fully SYNTHETIC business entity-resolution dataset.
No external data, no geocoding, no real business databases are used.

Usage:  python generate_dataset.py            (writes next to this script)
Output: dataset/train, dataset/test, evaluation/, ml/training_pairs.tsv, README.md
"""
import os, re, csv, json, copy, random, unicodedata
from collections import Counter, defaultdict
import numpy as np

random.seed(42)
np.random.seed(42)

ROOT = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# 1. VOCABULARY (all invented / generic)
# ----------------------------------------------------------------------------
CATS = {
    "restaurant":   ["Restaurant", "Kitchen", "Bistro", "Cafe", "Grill", "Biryani House", "Diner"],
    "hospital":     ["Hospital", "Medical Centre", "Clinic", "Healthcare", "Multispeciality Hospital"],
    "pharmacy":     ["Pharmacy", "Medicals", "Drugstore", "Chemists"],
    "supermarket":  ["Supermarket", "Hypermarket", "Mart", "Fresh Foods", "Grocers"],
    "technology":   ["Technologies", "Systems", "Tech Solutions", "Infotech", "Digital"],
    "software":     ["Software", "Softworks", "Labs", "Analytics", "Cloud"],
    "manufacturing":["Industries", "Manufacturing", "Steel Works", "Plastics", "Engineering", "Textiles"],
    "automobile":   ["Motors", "Auto", "Automobiles", "Cars", "Auto Sales"],
    "education":    ["Institute of Technology", "College", "University", "School", "Academy", "International School"],
    "hotel":        ["Hotel", "Inn", "Suites", "Resort", "Residency", "Grand Hotel"],
    "bank":         ["Bank", "Savings Bank", "Credit Union", "Cooperative Bank", "Finance"],
    "logistics":    ["Logistics", "Freight", "Cargo", "Transport", "Express", "Supply Chain"],
    "clothing":     ["Textiles", "Fashions", "Apparel", "Boutique", "Garments", "Clothing"],
    "electronics":  ["Electronics", "Digital World", "Gadgets", "Electricals", "Mobiles"],
    "construction": ["Constructions", "Builders", "Developers", "Infrastructure", "Engineering Works"],
    "consulting":   ["Consulting", "Consultants", "Advisors", "Associates", "Partners"],
}
CAT_W = {"restaurant": 9, "hospital": 5, "pharmacy": 6, "supermarket": 5, "technology": 8, "software": 8,
         "manufacturing": 7, "automobile": 5, "education": 6, "hotel": 6, "bank": 4, "logistics": 5,
         "clothing": 6, "electronics": 5, "construction": 6, "consulting": 6}
NOUN_LOCAL = {
    "France": {"restaurant": ["Restaurant", "Bistro", "Brasserie", "Cafe"], "pharmacy": ["Pharmacie"],
               "hotel": ["Hotel", "Auberge"], "bank": ["Banque", "Credit"], "supermarket": ["Supermarche", "Epicerie"]},
    "Germany": {"pharmacy": ["Apotheke"], "automobile": ["Autohaus", "Motors"], "restaurant": ["Restaurant", "Gasthaus"],
                "construction": ["Bau", "Bauunternehmen"], "bank": ["Bank", "Sparkasse"]},
}
TECHY = {"technology", "software", "manufacturing", "logistics", "construction", "consulting", "electronics"}
AMBIG = {"consulting", "construction", "bank", "clothing"}
LEGAL_P = {"technology": .9, "software": .9, "manufacturing": .9, "logistics": .85, "bank": .8, "consulting": .85,
           "construction": .85, "automobile": .8, "electronics": .7, "hospital": .5, "education": .4, "hotel": .55,
           "supermarket": .5, "clothing": .5, "restaurant": .25, "pharmacy": .3}
LEGAL_CHOICES = {
    "India": ["Private Limited"] * 5 + ["Limited"] * 2 + ["LLP"],
    "United States": ["Inc."] * 4 + ["LLC"] * 4 + ["Corporation", "Company"],
    "United Kingdom": ["Limited"] * 5 + ["PLC"],
    "Canada": ["Inc."] * 4 + ["Corporation", "Ltd."],
    "Australia": ["Pty Ltd"] * 5 + ["Limited"],
    "Singapore": ["Pte Ltd"] * 5 + ["Limited"],
    "Germany": ["GmbH"] * 5 + ["AG"],
    "France": ["SARL"] * 3 + ["SAS"] * 2 + ["SA"],
}
LEGAL_EQUIV = {
    "Private Limited": ["Pvt Ltd", "Pvt. Ltd.", "Pvt Ltd.", "Private Ltd", "P Ltd", "Pvt Limited"],
    "Limited": ["Ltd", "Ltd.", "Limited"], "Ltd.": ["Ltd", "Limited"],
    "Inc.": ["Inc", "Incorporated"], "Corporation": ["Corp", "Corp."], "Company": ["Co", "Co."],
    "LLC": ["L.L.C.", "LLC."], "LLP": ["L.L.P."], "PLC": ["Plc", "P.L.C."],
    "Pty Ltd": ["Pty. Ltd.", "Pty Limited", "Pty"], "Pte Ltd": ["Pte. Ltd.", "Pte Limited", "Private Limited"],
    "GmbH": ["G.m.b.H", "Gmbh", "mbH"], "AG": ["A.G."], "SARL": ["S.A.R.L.", "Sarl"], "SAS": ["S.A.S."], "SA": ["S.A."],
}
IN_WORDS = ["Lakshmi", "Annapoorna", "Vasanth", "Krishna", "Bharathi", "Murugan", "Aishwarya", "Sai Ram", "Kaveri",
            "Ganga", "Meenakshi", "Venkateswara", "Balaji", "Chola", "Pandian", "Kongu", "Saravana", "Ramanathan",
            "Vijaya", "Sundaram", "Thirumala", "Nandhini", "Ashoka", "Kalyani", "Gokul", "Padma", "Devi", "Karthik"]
US_WORDS = ["Summit", "Harbor", "Pioneer", "Lakeside", "Redwood", "Northstar", "Bluebird", "Ironwood", "Cedar Ridge",
            "Silverline", "Eagle", "Granite", "Beacon", "Crescent", "Evergreen", "Horizon", "Sterling", "Meridian",
            "Falcon", "Riverbend", "Oakmont", "Copperfield", "Windmill", "Stonebridge"]
UK_WORDS = ["Albion", "Thames", "Kingsway", "Ashford", "Wellington", "Brookside", "Hawthorn", "Northgate", "Fairview",
            "Blackwood", "Westbury", "Lancaster", "Harlow", "Pemberton", "Cranford", "Stratton"]
CA_WORDS = ["Maple Ridge", "Northern Lights", "Lakeshore", "Timberline", "Prairie", "Loon", "Glacier", "Cedar Point",
            "Birchwood", "Aurora", "Kootenay", "Fraser"]
AU_WORDS = ["Southern Cross", "Outback", "Coral", "Eucalypt", "Bayside", "Kestrel", "Wattle", "Banksia", "Harbourview",
            "Bluegum", "Ironbark", "Sunburst"]
SG_WORDS = ["Straits", "Orchid", "Jade", "Lotus", "Harbourfront", "Merchant", "Kopi", "Eastern Pearl", "Pacific Rim",
            "Tan & Lim", "Sentosa Bay", "Lion Gate"]
DE_WORDS = ["Schmidt", "Mueller", "Becker", "Hoffmann", "Wagner", "Weber", "Fischer", "Keller", "Krause", "Nordlicht",
            "Rheinland", "Bergmann", "Sonnenberg", "Waldmann"]
FR_WORDS = ["Lumiere", "Beaumont", "Dupont", "Lefevre", "Moreau", "Girard", "Rousseau", "Bonnet", "Laurent", "Fontaine",
            "Armand", "Perrin", "Soleil", "Belle Vue"]
# a few accented forms so accent-stripping noise is realistic
ACCENT_FIX = {"Lumiere": "Lumière", "Lefevre": "Lefèvre", "Mueller": "Müller", "Supermarche": "Supermarché",
              "Epicerie": "Épicerie", "Hotel": "Hôtel", "Credit": "Crédit"}
BRANDS = {"India": IN_WORDS, "United States": US_WORDS, "United Kingdom": UK_WORDS, "Canada": CA_WORDS,
          "Australia": AU_WORDS, "Singapore": SG_WORDS, "Germany": DE_WORDS, "France": FR_WORDS}
SURN = {"default": ["Harrison", "Mitchell", "Turner", "Bennett", "Cooper", "Reynolds", "Parker", "Hughes", "Foster", "Ellis",
                    "Sharma", "Iyer", "Nair", "Reddy", "Mehta", "Kapoor", "Tan", "Lim", "Wong", "Novak"],
        "Germany": ["Schmidt", "Mueller", "Becker", "Hoffmann", "Wagner", "Weber", "Fischer", "Keller"],
        "France": ["Dupont", "Lefevre", "Moreau", "Girard", "Rousseau", "Bonnet", "Laurent", "Fontaine"]}
MODS = ["Global", "Prime", "United", "Royal", "Metro", "Apex", "Pacific", "Central", "National", "Premier", "First",
        "Elite", "Alpha", "Nova", "Star"]
SYL = ["ka", "ra", "vi", "zen", "tri", "nov", "ax", "lum", "pri", "mon", "sol", "tek", "dyn", "vec", "nex", "qua", "ori",
       "el", "ist", "um", "an", "ta", "ri", "ma", "lo", "ve", "cor", "bit", "sys", "gen", "in", "fo", "ly", "do"]

# City tuple: (city, state_full, state_abbr, [postal prefixes])
CITIES = {
    "India": [("Chennai", "Tamil Nadu", "TN", ["600"]), ("Coimbatore", "Tamil Nadu", "TN", ["641"]),
              ("Madurai", "Tamil Nadu", "TN", ["625"]), ("Bengaluru", "Karnataka", "KA", ["560"]),
              ("Mumbai", "Maharashtra", "MH", ["400"]), ("Pune", "Maharashtra", "MH", ["411"]),
              ("New Delhi", "Delhi", "DL", ["110"]), ("Hyderabad", "Telangana", "TS", ["500"]),
              ("Kolkata", "West Bengal", "WB", ["700"]), ("Ahmedabad", "Gujarat", "GJ", ["380"]),
              ("Kochi", "Kerala", "KL", ["682"]), ("Jaipur", "Rajasthan", "RJ", ["302"])],
    "United States": [("New York", "New York", "NY", ["100"]), ("Austin", "Texas", "TX", ["787"]),
                      ("San Jose", "California", "CA", ["951"]), ("Chicago", "Illinois", "IL", ["606"]),
                      ("Seattle", "Washington", "WA", ["981"]), ("Boston", "Massachusetts", "MA", ["021"]),
                      ("Atlanta", "Georgia", "GA", ["303"]), ("Denver", "Colorado", "CO", ["802"]),
                      ("Miami", "Florida", "FL", ["331"]), ("Phoenix", "Arizona", "AZ", ["850"])],
    "United Kingdom": [("London", "England", "ENG", ["EC1", "SW3", "N7", "E14"]), ("Manchester", "England", "ENG", ["M2", "M4"]),
                       ("Leeds", "England", "ENG", ["LS1", "LS3"]), ("Birmingham", "England", "ENG", ["B1", "B4"]),
                       ("Edinburgh", "Scotland", "SCT", ["EH1", "EH2"])],
    "Canada": [("Toronto", "Ontario", "ON", ["M5V", "M4W"]), ("Vancouver", "British Columbia", "BC", ["V6B", "V5K"]),
               ("Calgary", "Alberta", "AB", ["T2P", "T3A"]), ("Montreal", "Quebec", "QC", ["H3B", "H2X"])],
    "Australia": [("Sydney", "New South Wales", "NSW", ["20"]), ("Melbourne", "Victoria", "VIC", ["30"]),
                  ("Brisbane", "Queensland", "QLD", ["40"]), ("Perth", "Western Australia", "WA", ["60"])],
    "Singapore": [("Singapore", "", "", ["00"])],
    "Germany": [("Berlin", "Berlin", "BE", ["10"]), ("Munich", "Bayern", "BY", ["80"]), ("Hamburg", "Hamburg", "HH", ["20"]),
                ("Frankfurt", "Hessen", "HE", ["60"]), ("Cologne", "Nordrhein-Westfalen", "NW", ["50"])],
    "France": [("Paris", "Ile-de-France", "IDF", ["75"]), ("Lyon", "Auvergne-Rhone-Alpes", "ARA", ["69"]),
               ("Marseille", "Provence-Alpes-Cote d'Azur", "PAC", ["13"]), ("Toulouse", "Occitanie", "OCC", ["31"]),
               ("Nice", "Provence-Alpes-Cote d'Azur", "PAC", ["06"])],
}
STREETS = {
    "India": (["MG", "Anna", "Gandhi", "Nehru", "Patel", "Tagore", "Kamaraj", "Bharathi", "Residency", "Temple", "Market",
               "Station", "Lake View", "Church", "Main", "Cross", "Ring", "Rajaji", "Subramaniam", "Vivekananda", "Ambedkar",
               "Race Course", "Mount", "Park", "Brigade", "Commercial", "Bazaar"],
              ["Road", "Street", "Salai", "Main Road", "Cross Street", "Avenue", "Lane"]),
    "United States": (["Maple", "Oak", "Main", "Market", "Broadway", "Pine", "Cedar", "Elm", "Washington", "Lincoln",
                       "Jefferson", "Park", "Lake", "Hillcrest", "Sunset", "Highland", "Mission", "Congress", "Commerce",
                       "5th", "42nd", "3rd"], ["Street", "Avenue", "Boulevard", "Drive", "Road", "Lane"]),
    "United Kingdom": (["High", "Church", "Station", "Victoria", "Albert", "King", "Queen", "Mill", "Bridge", "Castle",
                        "Park", "York", "Oxford"], ["Street", "Road", "Lane", "Avenue"]),
    "Canada": (["Yonge", "Queen", "King", "Granville", "Robson", "Bloor", "Portage", "Sherbrooke", "Maple", "Main"],
               ["Street", "Avenue", "Road", "Boulevard", "Drive"]),
    "Australia": (["George", "Collins", "Elizabeth", "Bourke", "Flinders", "Pitt", "Adelaide", "Hay", "Murray"],
                  ["Street", "Road", "Avenue", "Terrace"]),
    "Singapore": (["Orchard", "Bukit Timah", "Marina", "Tanjong Pagar", "Jurong", "Serangoon", "Changi", "Clementi"],
                  ["Road", "Street", "Avenue", "Drive"]),
    "Germany": (["Haupt", "Bahnhof", "Schiller", "Goethe", "Berliner", "Linden", "Garten", "Schul", "Kirch", "Markt"],
                ["straße", "straße", "straße", "allee", "weg"]),
    "France": (["Victor Hugo", "de la Paix", "Gambetta", "Jean Jaures", "de la Republique", "Pasteur", "Voltaire",
                "Lafayette", "du Commerce", "Saint-Michel"], ["Rue", "Rue", "Avenue", "Boulevard", "Place"]),
}
AREA_PRE = ["Anna", "Gandhi", "Rajaji", "Lakshmi", "Vijaya", "Saraswathi", "Shivaji", "Krishna", "Indira", "Sardar",
            "Ashok", "Kamala", "Ganesh", "Vasant", "Mahatma"]
AREA_SUF = ["Nagar", "Colony", "Layout", "Extension", "Puram", "Vihar", "Enclave", "Park"]
LM_KINDS = {"India": ["Bus Stand", "Temple", "Railway Station", "Mall", "Signal", "Park", "Bus Stop"]}
LM_KINDS_DEFAULT = ["Station", "Mall", "Park", "Library", "Plaza", "Square"]
LM_TPL = ["Near {}", "Opp. {}", "Behind {}", "Next to {}", "Close to {}", "Nr. {}"]
UNIT_T = ["Building {L}", "Apartment {n}", "Floor {n}", "Suite {nnn}", "Tower {L}", "Block {L}"]
ABBR_UNIT = {"Apartment": "Apt", "Building": "Bldg", "Floor": "Fl", "Suite": "Ste"}
ABBR_TYPE = {"Road": "Rd", "Street": "St", "Avenue": "Ave", "Boulevard": "Blvd", "Drive": "Dr", "Lane": "Ln",
             "Terrace": "Tce", "Main Road": "Main Rd", "Cross Street": "Cross St", "Place": "Pl", "Rue": "R.",
             "allee": "Al.", "Salai": "Sl"}
STATE_DEFAULT = {"India": "full", "United States": "abbr", "Canada": "abbr", "Australia": "abbr"}
COUNTRY_VARIANTS = {
    "India": ["India", "INDIA", "IN", "Bharat"], "United States": ["United States", "USA", "US", "U.S.A.", "United States of America"],
    "United Kingdom": ["United Kingdom", "UK", "U.K.", "Great Britain"], "Canada": ["Canada", "CA", "CAN"],
    "Australia": ["Australia", "AU", "AUS"], "Singapore": ["Singapore", "SG", "SGP"],
    "Germany": ["Germany", "DE", "Deutschland", "GERMANY"], "France": ["France", "FR", "FRANCE"],
}
COUNTRY_CANON = {}
for _c, _vs in COUNTRY_VARIANTS.items():
    for _v in _vs:
        COUNTRY_CANON[_v.upper()] = _c

ABBR_PAIRS = {"Technologies": "Tech", "Technology": "Tech", "International": "Intl", "Engineering": "Engg",
              "Hospital": "Hosp", "Supermarket": "Supermkt", "Constructions": "Constr", "Consultants": "Consult",
              "Logistics": "Logis", "Industries": "Inds", "Manufacturing": "Mfg", "Automobiles": "Autos",
              "Electronics": "Electr", "Institute": "Inst", "University": "Univ", "Restaurant": "Rest.",
              "Pharmacy": "Pharm", "Solutions": "Sol", "Systems": "Sys", "Software": "SW", "Associates": "Assoc",
              "Services": "Svcs", "Limited": "Ltd", "Private": "Pvt", "Corporation": "Corp", "Company": "Co",
              "Incorporated": "Inc", "Saint": "St", "Sri": "Shri", "Multispeciality": "Multi-Spl", "Developers": "Devs",
              "Residency": "Resid.", "Cooperative": "Coop", "Apparel": "Appar."}
ABBR_REV = {v: k for k, v in ABBR_PAIRS.items() if v in ("Tech", "Ltd", "Pvt", "Corp", "Co", "Inc", "Intl", "Engg", "Univ", "Inst")}
MISSPELL = {"Technologies": ["Tecnologies", "Technologeis", "Technolgies"], "Restaurant": ["Restarant", "Resturant"],
            "Pharmacy": ["Pharmacey", "Farmacy"], "Hospital": ["Hospitel", "Hosptial"], "Supermarket": ["Supermarkit", "Super Market"],
            "Industries": ["Industires", "Indusries"], "Engineering": ["Enginering", "Engneering"],
            "Consulting": ["Consultng", "Consluting"], "Logistics": ["Logistcs", "Logisitcs"], "Electronics": ["Electrnics"],
            "Software": ["Softwear", "Sofware"], "Constructions": ["Constructons", "Contructions"],
            "Institute": ["Institue", "Insitute"], "University": ["Univercity", "Universty"], "Solutions": ["Solutons", "Soltions"],
            "Academy": ["Acadamy", "Academi"], "Manufacturing": ["Manufactring", "Manufacuring"], "Automobiles": ["Automobils"]}
TRANSLIT = [("v", "w"), ("ee", "i"), ("sh", "s"), ("th", "t"), ("oo", "u"), ("ph", "f"), ("aa", "a"), ("Lakshmi", "Laxmi"),
            ("Krishna", "Krishnaa"), ("Venkateswara", "Venkateshwara"), ("Sri", "Shri"), ("Murugan", "Murugaan"),
            ("Meenakshi", "Minakshi"), ("Bharathi", "Bharati"), ("Saravana", "Saravanaa"), ("Vasanth", "Vasant"),
            ("ck", "k"), ("ou", "u"), ("y", "i"), ("z", "s"), ("Mueller", "Muller"), ("Schmidt", "Schmitt")]
KEYB = {'a': 'qsz', 'b': 'vgn', 'c': 'xvd', 'd': 'sfe', 'e': 'wrd', 'f': 'dgr', 'g': 'fht', 'h': 'gjy', 'i': 'uok',
        'j': 'hkn', 'k': 'jlm', 'l': 'kop', 'm': 'nk', 'n': 'bhm', 'o': 'ipl', 'p': 'ol', 'q': 'wa', 'r': 'etf',
        's': 'adw', 't': 'ryg', 'u': 'yij', 'v': 'cfb', 'w': 'qes', 'x': 'zcs', 'y': 'tuh', 'z': 'xas'}
LETTERS = "ABCDEFGHJKLMNPRSTUWXY"

# ----------------------------------------------------------------------------
# 2. HELPERS
# ----------------------------------------------------------------------------
def wchoice(rng, items):
    return rng.choices([v for v, _ in items], weights=[w for _, w in items], k=1)[0]

def wsample(rng, items, k):
    items, out = list(items), []
    for _ in range(min(k, len(items))):
        v = wchoice(rng, items)
        out.append(v)
        items = [(a, b) for a, b in items if a != v]
    return out

def strip_accents(s):
    s = s.replace("ß", "ss")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def typo_word(w, rng):
    if len(w) < 4 or not w.isalpha():
        return w
    i = rng.randrange(1, len(w) - 1)
    op = rng.choice(["swap", "del", "dup", "sub"])
    if op == "swap":
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    if op == "del":
        return w[:i] + w[i + 1:]
    if op == "dup":
        return w[:i] + w[i] + w[i:]
    r = KEYB.get(w[i].lower())
    if r:
        n = rng.choice(r)
        return w[:i] + (n.upper() if w[i].isupper() else n) + w[i + 1:]
    return w

def sanitize(s):
    return re.sub(r"[\t\r\n\"]", " ", s)

# ----------------------------------------------------------------------------
# 3. BUSINESS / ADDRESS GENERATION
# ----------------------------------------------------------------------------
def coined(rng):
    w = "".join(rng.choice(SYL) for _ in range(rng.choice([2, 2, 3]))).capitalize()
    if rng.random() < .35:
        w += rng.choice(["ix", "ex", "a", "on", "ia"])
    return w

def gen_brand(country, cat, rng):
    r = rng.random()
    techy = cat in TECHY
    if techy and r < .12:
        return "".join(rng.choice("ABCDEFGHKLMNPRSTVX") for _ in range(3))
    if country == "India" and r < .45:
        w = rng.choice(IN_WORDS)
        if rng.random() < .3:
            w = rng.choice(["Sri", "Shree"]) + " " + w
        return w
    if (techy and r < .55) or r < .25:
        return coined(rng)
    if cat in AMBIG and r < .55:
        a, b = rng.sample(SURN.get(country, SURN["default"]), 2)
        return f"{a} & {b}"
    w = rng.choice(BRANDS[country])
    if rng.random() < .3:
        w = rng.choice(MODS) + " " + w
    return w

def gen_pin(country, prefix, rng):
    if country == "India":
        return prefix + "%03d" % rng.randrange(1000)
    if country == "United States":
        return prefix + "%02d" % rng.randrange(100)
    if country == "United Kingdom":
        return prefix + " " + str(rng.randrange(1, 10)) + rng.choice(LETTERS) + rng.choice(LETTERS)
    if country == "Canada":
        return prefix + " " + str(rng.randrange(1, 10)) + rng.choice(LETTERS) + str(rng.randrange(1, 10))
    if country == "Australia":
        return prefix + "%02d" % rng.randrange(100)
    if country == "Singapore":
        return "%02d%04d" % (rng.randrange(1, 82), rng.randrange(10000))
    return prefix + "%03d" % rng.randrange(1000)  # DE / FR

def gen_address(country, rng, city_t=None, same_street_of=None):
    ct = city_t or rng.choice(CITIES[country])
    names, types = STREETS[country]
    bno = {"India": lambda: str(rng.randint(1, 250)) + rng.choice(["", "", "", "A", "B", "/1"]),
           "United States": lambda: str(rng.randint(100, 9999)),
           "Germany": lambda: str(rng.randint(1, 120)) + rng.choice(["", "", "a"])}.get(
        country, lambda: str(rng.randint(1, 300)))()
    a = dict(country=country, bno=bno, sname=rng.choice(names), stype=rng.choice(types), area="",
             city=ct[0], state=ct[1], sabbr=ct[2], pin=gen_pin(country, rng.choice(ct[3]), rng),
             pin4="%04d" % rng.randrange(10000), unit="", lm="", _ct=ct)
    if same_street_of:
        a["sname"], a["stype"] = same_street_of["sname"], same_street_of["stype"]
    if country == "India":
        a["area"] = rng.choice(AREA_PRE) + " " + rng.choice(AREA_SUF)
    if rng.random() < .3:
        t = rng.choice(UNIT_T)
        a["unit"] = t.format(L=rng.choice("ABCDE"), n=rng.randint(1, 12), nnn=rng.randint(101, 950))
    kinds = LM_KINDS.get(country, LM_KINDS_DEFAULT)
    a["lm"] = rng.choice(names).replace("straße", "") + " " + rng.choice(kinds)
    return a

def gen_business(country, cat, rng, city_t=None):
    noun = rng.choice(NOUN_LOCAL.get(country, {}).get(cat, CATS[cat]))
    legal = rng.choice(LEGAL_CHOICES[country]) if rng.random() < LEGAL_P[cat] else ""
    brand = gen_brand(country, cat, rng)
    brand = ACCENT_FIX.get(brand, brand)
    noun = ACCENT_FIX.get(noun, noun)
    return dict(cat=cat, country=country, brand=brand, noun=noun, legal=legal,
                addr=gen_address(country, rng, city_t), origin="base", clone_of=None)

def full_name(b):
    return " ".join(x for x in [b["brand"], b["noun"], b["legal"]] if x)

def name_variant(b, rng):
    """A DIFFERENT business whose name resembles b's."""
    b2 = copy.deepcopy(b)
    alt = [n for n in CATS[b["cat"]] if n != b["noun"]]
    r = rng.random()
    if r < .4 and alt:
        b2["noun"] = rng.choice(alt)
    elif r < .7:
        b2["brand"] = b["brand"] + " " + rng.choice(["Plus", "One", "Prime", "Global", "International", "Group"])
    elif r < .85:
        w = b["brand"]
        b2["brand"] = (w[:-1] + rng.choice("aeiouxs")) if len(w) > 3 and w[-1] not in "aeiouxs" else w + "x"
    else:
        others = [l for l in LEGAL_CHOICES[b["country"]] if l != b["legal"]]
        if others:
            b2["legal"] = rng.choice(others)
        else:
            b2["noun"] = rng.choice(alt) if alt else b["noun"]
    if full_name(b2) == full_name(b):
        b2["brand"] += " Co"
    return b2

def gen_s1_masters(cfg, rng):
    weights = cfg["country_w"]
    countries = list(weights)
    cw = [weights[c] for c in countries]
    cats = list(CAT_W)
    catw = [CAT_W[c] for c in cats]
    n_base = cfg["n_s1"] - cfg["n_chain"] - cfg["n_sib"]
    used, masters = set(), []

    def key(b):
        return (full_name(b).lower(), render(b["addr"], {}, rng).lower())

    def add(b):
        k = key(b)
        if k in used:
            return False
        used.add(k)
        masters.append(b)
        return True

    while len(masters) < n_base:
        c = rng.choices(countries, cw)[0]
        add(gen_business(c, rng.choices(cats, catw)[0], rng))
    base_list = list(masters)
    # chain branches: identical name, different address (hard negatives by construction)
    tries = 0
    target = n_base + cfg["n_chain"]
    while len(masters) < target and tries < 10000:
        tries += 1
        b = copy.deepcopy(rng.choice(base_list))
        same_city = rng.random() < .4
        ct = b["addr"]["_ct"] if same_city else rng.choice(CITIES[b["country"]])
        b["addr"] = gen_address(b["country"], rng, ct)
        b["origin"] = "chain_branch"
        add(b)
    # siblings: similar names in same city, or similar address with different name
    tries = 0
    while len(masters) < cfg["n_s1"] and tries < 10000:
        tries += 1
        base = rng.choice(base_list)
        if rng.random() < .5:
            b = name_variant(base, rng)
            b["addr"] = gen_address(b["country"], rng, base["addr"]["_ct"])
            b["origin"] = "sibling_similar_name"
        else:
            b = gen_business(base["country"], base["cat"], rng, base["addr"]["_ct"])
            b["addr"] = gen_address(b["country"], rng, base["addr"]["_ct"], same_street_of=base["addr"])
            b["addr"]["bno"] = str(max(1, int(re.sub(r"\D", "", base["addr"]["bno"]) or 1) + rng.choice([-4, -2, 2, 4, 6])))
            b["origin"] = "sibling_similar_address"
        add(b)
    rng.shuffle(masters)
    return masters

def gen_decoy(kind, s1, rng):
    """Business that is NOT in Source 1 but is confusable with one that is."""
    base = rng.choice(s1)
    if kind == "same_name_diff_addr":
        b = copy.deepcopy(base)
        ct = base["addr"]["_ct"] if rng.random() < .5 else rng.choice(CITIES[base["country"]])
        b["addr"] = gen_address(b["country"], rng, ct)
    elif kind == "similar_name_diff_addr":
        b = name_variant(base, rng)
        ct = base["addr"]["_ct"] if rng.random() < .5 else rng.choice(CITIES[base["country"]])
        b["addr"] = gen_address(b["country"], rng, ct)
    elif kind == "same_city_diff_biz":
        b = gen_business(base["country"], base["cat"], rng, base["addr"]["_ct"])
    elif kind == "similar_addr_diff_name":
        b = gen_business(base["country"], rng.choice(list(CAT_W)), rng, base["addr"]["_ct"])
        b["addr"] = gen_address(b["country"], rng, base["addr"]["_ct"], same_street_of=base["addr"])
        b["addr"]["bno"] = base["addr"]["bno"] if rng.random() < .5 else str(rng.randint(1, 300))
    else:
        c = rng.choice(list(BRANDS))
        b = gen_business(c, rng.choice(list(CAT_W)), rng)
    b["origin"] = "decoy_" + kind
    b["clone_of"] = base["_mid"] if kind != "random" else None
    return b

DECOY_KINDS = [("same_name_diff_addr", .35), ("similar_name_diff_addr", .25), ("same_city_diff_biz", .15),
               ("similar_addr_diff_name", .15), ("random", .10)]

# ----------------------------------------------------------------------------
# 4. ADDRESS RENDERING
# ----------------------------------------------------------------------------
def fmt_bno(bno, style):
    if style == "plain" or not bno: return bno
    if style == "No.": return "No. " + bno
    if style == "No": return "No " + bno
    if style == "#": return "#" + bno
    if style == "dash": return bno.replace("/", "-")
    if style == "space": return re.sub(r"(\d)([A-Z])", r"\1 \2", bno)
    return bno

def render(a, o, rng):
    c = a["country"]
    st = a["stype"]
    bno = fmt_bno("" if o.get("drop_bno") else a["bno"], o.get("bno_style", "plain"))
    sname = a["sname"]
    if o.get("dotted"):
        f = (lambda t: ".".join(t) + ".") if o["dotted"] == "dots" else (lambda t: " ".join(t))
        sname = " ".join(f(t) if t.isupper() and 2 <= len(t) <= 3 else t for t in sname.split())
    if o.get("abbr"):
        if c == "Germany":
            st = st.replace("straße", "str.")
        else:
            st = ABBR_TYPE.get(st, st)
            if o.get("abbr_period") and not st.endswith("."):
                st += "."
    if o.get("drop_street"): line1 = ""
    elif c == "Germany": line1 = f"{sname}{st} {bno}".strip()
    elif c == "France": line1 = f"{bno} {st} {sname}".strip()
    else: line1 = f"{bno} {sname} {st}".strip()
    unit = "" if o.get("drop_unit") else a.get("unit", "")
    if unit and o.get("abbr"):
        for k, v in ABBR_UNIT.items():
            unit = re.sub("^" + k, v, unit)
    extra = ""
    lm = o.get("landmark")
    if lm == "replace": line1 = o["lm_tpl"].format(a["lm"])
    elif lm == "append": extra = o["lm_tpl"].format(a["lm"])
    elif lm == "street_only": line1 = "Near " + f"{sname} {st}".strip()
    # locality
    style = o.get("state_style") or STATE_DEFAULT.get(c)
    state = ""
    if o.get("use_state", True) and a["state"] and c in STATE_DEFAULT:
        state = a["sabbr"] if style == "abbr" else a["state"]
    pin = a["pin"] if o.get("use_pin", True) else ""
    pf = o.get("pinfmt")
    if pin:
        if pf == "compact": pin = pin.replace(" ", "")
        elif pf == "space" and c == "India": pin = pin[:3] + " " + pin[3:]
        elif pf == "zip4" and c == "United States": pin = pin + "-" + a["pin4"]
        if o.get("pin_prefix"):
            pin = {"India": "PIN ", "United States": "ZIP ", "Canada": "Postal Code ", "United Kingdom": "Postcode "}.get(c, "") + pin
    city = a["city"] if o.get("use_city", True) else ""
    area = a["area"] if (c == "India" and o.get("use_area", True)) else ""
    loc = []
    if c == "India":
        loc = [area, city, " ".join(x for x in [state, pin] if x)]
    elif c in ("United States", "Canada"):
        loc = [city, " ".join(x for x in [state, pin] if x)]
    elif c == "Australia":
        loc = [" ".join(x for x in [city, state, pin] if x)]
    elif c in ("United Kingdom", "Singapore"):
        loc = [" ".join(x for x in [city, pin] if x)] if c == "Singapore" else [city, pin]
    else:  # DE / FR
        loc = [" ".join(x for x in [pin, city] if x)]
    first = [unit, line1, extra] if o.get("unit_lead") else [line1, unit, extra]
    if o.get("reorder") == "street_last": segs = loc + first
    elif o.get("reorder") == "rev": segs = (first + loc)[::-1]
    else: segs = first + loc
    segs = [s for s in segs if s]
    if o.get("append_country"): segs.append(o["append_country"])
    s = o.get("sep", ", ").join(segs)
    if o.get("punct_strip"): s = re.sub(r"\s+", " ", re.sub(r"[.,]", "", s))
    if o.get("accent"): s = strip_accents(s)
    if o.get("upper"): s = s.upper()
    if o.get("lower"): s = s.lower()
    if o.get("spaces") and " " in s:
        idx = [m.start() for m in re.finditer(" ", s)]
        i = rng.choice(idx)
        s = s[:i] + "  " + s[i + 1:]
    return s.strip() if not o.get("spaces") else s

# ----------------------------------------------------------------------------
# 5. NOISE ENGINE
# ----------------------------------------------------------------------------
PROFILE_W = {
    "train": {"S2": [("exact", .20), ("abbrev", .34), ("typo", .20), ("address", .06), ("missing", .07), ("order", .09), ("complex", .04)],
              "S3": [("exact", .20), ("abbrev", .14), ("typo", .08), ("address", .26), ("missing", .13), ("order", .12), ("complex", .07)]},
    "test":  {"S2": [("exact", .12), ("abbrev", .26), ("typo", .16), ("address", .14), ("missing", .10), ("order", .12), ("complex", .10)],
              "S3": [("exact", .12), ("abbrev", .14), ("typo", .10), ("address", .24), ("missing", .14), ("order", .14), ("complex", .12)]},
}
ALL_NAME_OPS = [("legal_variant", 2), ("legal_drop", 1), ("shorten", 1.5), ("drop_word", 1), ("initials", 1), ("abbrev", 3),
                ("translit", 1.5), ("typo", 3), ("reorder", 1.5), ("punct", 2), ("case", 2), ("spaces", 1)]
ALL_ADDR_OPS = [("abbr", 2), ("state_abbr", 1), ("drop_pin", 2), ("drop_state", 1), ("drop_city", .5), ("drop_bno", 1),
                ("drop_unit", 1), ("landmark", 2), ("reorder", 1.5), ("punct", 1.5), ("sep", 1), ("numstyle", 1),
                ("dotted", 1), ("typo", 1.5), ("pinfmt", 1), ("upper", .5), ("spaces", .5), ("accent", .5)]
TEST_ONLY_NAME = ["the_prefix", "vowel_drop", "group_suffix", "umlaut_expand"]
TEST_ONLY_ADDR = ["pin_prefix", "unit_lead", "append_country"]

def choose_ops(prof, src, split, rng):
    n, a, cm = [], [], None
    if prof == "exact":
        n = [] if rng.random() < .7 else [rng.choice(["case", "punct"])]
        a = [] if rng.random() < .7 else [rng.choice(["punct", "abbr"])]
    elif prof == "abbrev":
        pool = ([("legal_variant", 3), ("abbrev", 4), ("initials", 1), ("shorten", .5)] if src == "S2" else
                [("legal_variant", 1), ("abbrev", 2), ("shorten", 3), ("initials", 1), ("legal_drop", 1)])
        n = wsample(rng, pool, rng.choice([1, 2]))
        if "abbrev" not in n and rng.random() < .5: n.append("abbrev")
        a = ["abbr"] if rng.random() < .6 else []
        if rng.random() < .3: a.append("state_abbr")
    elif prof == "typo":
        n = ["typo"]
        if src == "S3" and rng.random() < .5: n.append("translit")
        if rng.random() < .3: n.append("legal_variant")
        a = ["typo"] if rng.random() < .3 else []
    elif prof == "address":
        n = [] if rng.random() < .4 else [rng.choice(["legal_variant", "case"])]
        pool = ([("abbr", 3), ("state_abbr", 1), ("drop_pin", 1), ("punct", 2), ("numstyle", 2), ("sep", 1), ("dotted", 1), ("drop_unit", 1)]
                if src == "S2" else
                [("drop_pin", 3), ("drop_state", 2), ("drop_city", 1), ("landmark", 3), ("reorder", 2), ("drop_bno", 1.5),
                 ("drop_unit", 2), ("pinfmt", 1), ("abbr", 1)])
        a = wsample(rng, pool, rng.choice([1, 2, 3]))
    elif prof == "missing":
        r = rng.random()
        if r < .30: a = ["missing_addr"]
        elif r < .55: cm = "missing"
        elif r < .80: a = ["drop_street", "drop_pin", "drop_unit"]
        else: a = ["drop_pin", "drop_state"]
        if rng.random() < .4: n = [rng.choice(["drop_word", "legal_drop", "shorten"])]
    elif prof == "order":
        pool = ([("punct", 3), ("case", 3), ("spaces", 2), ("reorder", 1)] if src == "S2" else
                [("reorder", 3), ("punct", 2), ("case", 2), ("spaces", 1)])
        n = wsample(rng, pool, 2)
        a = wsample(rng, [("punct", 2), ("sep", 2), ("upper", 1), ("spaces", 1)], rng.choice([0, 1, 2]))
        if src == "S3" and rng.random() < .3: a.append("reorder")
    else:  # complex
        n = wsample(rng, ALL_NAME_OPS, rng.choice([2, 3, 4]))
        a = wsample(rng, ALL_ADDR_OPS, rng.choice([2, 3, 4]))
        if rng.random() < .25: cm = rng.choice(["variant", "missing"])
    if split == "test" and prof != "exact":
        if rng.random() < .35: n.append(rng.choice(TEST_ONLY_NAME))
        if rng.random() < .30: a.append(rng.choice(TEST_ONLY_ADDR))
    # background missingness
    if "missing_addr" not in a:
        if rng.random() < (.02 if src == "S2" else .04): a = ["missing_addr"]
        elif "drop_pin" not in a and rng.random() < (.05 if src == "S2" else .12): a.append("drop_pin")
    return n, a, cm

def build_name(b, ops, rng):
    brand, noun, legal = b["brand"], b["noun"], b["legal"]
    order = None
    for op in ops:
        if op == "legal_variant":
            if legal in LEGAL_EQUIV: legal = rng.choice(LEGAL_EQUIV[legal])
        elif op == "legal_drop": legal = ""
        elif op == "shorten":
            t = noun.split()
            noun = t[-1] if len(t) > 1 and rng.random() < .5 else ""
        elif op == "drop_word":
            t = brand.split()
            if len(t) >= 2:
                del t[rng.randrange(len(t))]; brand = " ".join(t)
            elif len(noun.split()) > 1: noun = noun.split()[-1]
        elif op == "initials":
            t = brand.split()
            if len(t) >= 2 and "&" not in t: brand = ".".join(x[0] for x in t) + "." if rng.random() < .5 else "".join(x[0] for x in t)
            elif brand.isupper() and len(brand) <= 4: brand = rng.choice([".".join(brand), " ".join(brand)])
        elif op == "reorder": order = rng.choice(["noun_first", "comma"])
        elif op == "group_suffix": noun = (noun + " Group").strip()
    if not noun: s = brand
    elif order == "noun_first": s = f"{noun} {brand}"
    elif order == "comma": s = f"{noun}, {brand}"
    else: s = f"{brand} {noun}"
    if legal: s += " " + legal
    for op in ops:
        if op == "abbrev":
            toks = s.split(" ")
            for i, t in enumerate(toks):
                if t == "&" and rng.random() < .6: toks[i] = "and"
                elif t.lower() == "and" and rng.random() < .3: toks[i] = "&"
                elif t in ABBR_PAIRS and rng.random() < .7: toks[i] = ABBR_PAIRS[t]
                elif t in ABBR_REV and rng.random() < .3: toks[i] = ABBR_REV[t]
            s = " ".join(toks)
        elif op == "translit":
            if re.search(r"[^\x00-\x7f]", s): s = strip_accents(s)
            else:
                pairs = TRANSLIT[:]; rng.shuffle(pairs)
                for x, y in pairs:
                    for p, q in ((x, y), (y, x)):
                        m = re.search(re.escape(p), s, re.I)
                        if m and len(s) > 6:
                            rep = q.capitalize() if m.group(0)[0].isupper() else q
                            s = s[:m.start()] + rep + s[m.end():]
                            break
                    else:
                        continue
                    break
        elif op == "typo":
            toks = s.split(" ")
            cand = [i for i, t in enumerate(toks) if len(t) >= 4 and t.isalpha()]
            if cand:
                for i in rng.sample(cand, min(len(cand), rng.choice([1, 1, 2]))):
                    ms = MISSPELL.get(toks[i])
                    toks[i] = rng.choice(ms) if ms and rng.random() < .5 else typo_word(toks[i], rng)
            s = " ".join(toks)
        elif op == "vowel_drop":
            toks = s.split(" ")
            cand = [i for i, t in enumerate(toks) if len(t) >= 7 and t.isalpha()]
            if cand:
                i = rng.choice(cand); toks[i] = toks[i][0] + re.sub(r"[aeiou]", "", toks[i][1:])
            s = " ".join(toks)
        elif op == "umlaut_expand":
            s = s.replace("ü", "ue").replace("ö", "oe").replace("ä", "ae").replace("Mueller", "Müller") if rng.random() < .5 else strip_accents(s)
        elif op == "the_prefix":
            s = "The " + s
        elif op == "punct":
            s = s.replace(" & ", rng.choice(["&", " and "]))
            s = re.sub(r"[.,]", "", s).replace("-", " ")
            s = re.sub(r"\s+", " ", s).strip()
        elif op == "case":
            s = rng.choices([s.upper(), s.lower(), s.title()], [.5, .2, .3])[0]
        elif op == "spaces":
            toks = s.split(" "); i = rng.randrange(len(toks)); toks[i] += " "
            s = " ".join(toks)
            if rng.random() < .3: s += " "
    return s

def build_addr(b, aops, rng):
    a = b["addr"]
    if "missing_addr" in aops: return "", dict(missing_addr=True)
    o = {}
    a2 = dict(a)
    for op in aops:
        if op == "abbr": o["abbr"] = True; o["abbr_period"] = rng.random() < .3
        elif op == "state_abbr": o["state_style"] = "abbr"
        elif op == "drop_pin": o["use_pin"] = False
        elif op == "drop_state": o["use_state"] = False
        elif op == "drop_city": o["use_city"] = False
        elif op == "drop_bno": o["drop_bno"] = True
        elif op == "drop_unit": o["drop_unit"] = True
        elif op == "drop_street": o["drop_street"] = True
        elif op == "landmark": o["landmark"] = rng.choice(["replace", "append", "street_only"]); o["lm_tpl"] = rng.choice(LM_TPL)
        elif op == "reorder": o["reorder"] = rng.choice(["rev", "street_last"])
        elif op == "punct": o["punct_strip"] = True; o["sep"] = " "
        elif op == "sep": o["sep"] = rng.choice([" ", ",", " - ", "; "])
        elif op == "numstyle": o["bno_style"] = rng.choice(["No.", "#", "No", "dash", "space"])
        elif op == "dotted": o["dotted"] = rng.choice(["dots", "spaced"])
        elif op == "pinfmt": o["pinfmt"] = rng.choice(["compact", "space", "zip4"])
        elif op == "upper": o["upper"] = True
        elif op == "spaces": o["spaces"] = True
        elif op == "accent": o["accent"] = True
        elif op == "pin_prefix": o["pin_prefix"] = True
        elif op == "unit_lead": o["unit_lead"] = True
        elif op == "append_country": o["append_country"] = rng.choice(COUNTRY_VARIANTS[a["country"]])
        elif op == "typo":
            if rng.random() < .6: a2["sname"] = " ".join(typo_word(t, rng) for t in a["sname"].split())
            else: a2["city"] = typo_word(a["city"], rng)
    s = render(a2, o, rng)
    return s, dict(missing_addr=False, pin_missing=not o.get("use_pin", True) or bool(o.get("landmark") == "replace"))

def make_record(biz, src, split, rng):
    prof = wchoice(rng, PROFILE_W[split][src])
    nops, aops, cm = choose_ops(prof, src, split, rng)
    name = build_name(biz, nops, rng)
    addr, flags = build_addr(biz, aops, rng)
    pm = .03 if src == "S2" else .06
    pv = .15 if src == "S2" else .25
    if cm is None:
        r = rng.random()
        cm = "missing" if r < pm else ("variant" if r < pm + pv else "canon")
    country = "" if cm == "missing" else (rng.choice(COUNTRY_VARIANTS[biz["country"]]) if cm == "variant" else biz["country"])
    return dict(name=sanitize(name), address=sanitize(addr), country=country, profile=prof,
                name_ops=nops, addr_ops=aops, country_mode=cm, flags=flags)

# ----------------------------------------------------------------------------
# 6. SPLIT GENERATION
# ----------------------------------------------------------------------------
CFG = {
    "train": dict(seed=42, n_s1=2000, n_chain=150, n_sib=110, single_frac=.15, t2=3000, t3=3000,
                  country_w={"India": .38, "United States": .27, "United Kingdom": .08, "Canada": .07, "Australia": .06,
                             "Germany": .05, "France": .05, "Singapore": .04}),
    "test":  dict(seed=2024, n_s1=600, n_chain=50, n_sib=40, single_frac=.18, t2=900, t3=900,
                  country_w={"India": .25, "United States": .20, "United Kingdom": .10, "Canada": .08, "Australia": .08,
                             "Germany": .11, "France": .10, "Singapore": .08}),
}
COUNT_W = [(0, .15), (1, .35), (2, .30), (3, .15), (4, .05)]

def gen_split(split):
    cfg = CFG[split]
    rng = random.Random(cfg["seed"])
    masters = gen_s1_masters(cfg, rng)
    for i, m in enumerate(masters):
        m["_mid"] = "M%06d" % (i + 1)
    singles = set(rng.sample(range(len(masters)), int(len(masters) * cfg["single_frac"])))
    recs = {"S2": [], "S3": []}   # (biz, master_mid, kind)
    for i, m in enumerate(masters):
        if i in singles: continue
        while True:
            n2, n3 = wchoice(rng, COUNT_W), wchoice(rng, COUNT_W)
            if n2 + n3 > 0: break
        recs["S2"] += [(m, m["_mid"], "match")] * n2
        recs["S3"] += [(m, m["_mid"], "match")] * n3
    decoys, decoy_id = [], 0
    def new_decoy():
        nonlocal decoy_id
        d = gen_decoy(wchoice(rng, DECOY_KINDS), masters, rng)
        decoy_id += 1
        d["_mid"] = "D%06d" % decoy_id
        return d
    need2 = max(int(.05 * cfg["t2"]), cfg["t2"] - len(recs["S2"]))
    d2 = [new_decoy() for _ in range(need2)]
    need3 = max(int(.05 * cfg["t3"]), cfg["t3"] - len(recs["S3"]))
    reuse = rng.sample(d2, min(int(.4 * need3), len(d2)))
    d3 = reuse + [new_decoy() for _ in range(need3 - len(reuse))]
    recs["S2"] += [(d, d["_mid"], d["origin"]) for d in d2]
    recs["S3"] += [(d, d["_mid"], d["origin"]) for d in d3]
    # independent ID assignment (shuffled)
    order = list(range(len(masters))); rng.shuffle(order)
    s1_rows, mid2s1 = [], {}
    for k, i in enumerate(order):
        m = masters[i]
        sid = "S1-%05d" % (k + 1)
        mid2s1[m["_mid"]] = sid
        s1_rows.append(dict(entity_id=sid, business_name=sanitize(full_name(m)), business_address=render(m["addr"], {}, rng),
                            country=m["country"], master=m["_mid"], origin=m["origin"]))
    out = {"S1": s1_rows}
    for src in ("S2", "S3"):
        lst = recs[src]; rng.shuffle(lst)
        rows = []
        for k, (biz, mid, kind) in enumerate(lst):
            r = make_record(biz, src, split, rng)
            r.update(entity_id="%s-%05d" % (src, k + 1), master=mid, kind=kind)
            rows.append(r)
        out[src] = rows
    gt = defaultdict(list)
    for src in ("S2", "S3"):
        for r in out[src]:
            if r["master"] in mid2s1: gt[mid2s1[r["master"]]].append(r["entity_id"])
    gt_rows = [(r["entity_id"], sorted(gt.get(r["entity_id"], []))) for r in s1_rows]
    return out, gt_rows, mid2s1

# ----------------------------------------------------------------------------
# 7. SIMILARITY FEATURES (pure python, deterministic)
# ----------------------------------------------------------------------------
def norm(s):
    if not s: return ""
    s = strip_accents(s).lower().replace("&", " and ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()

def lcs_len(a, b):
    if not a or not b: return 0
    masks = {}
    for i, ch in enumerate(a): masks[ch] = masks.get(ch, 0) | (1 << i)
    m = len(a); full = (1 << m) - 1; V = full
    for ch in b:
        M = masks.get(ch, 0); U = V & M
        V = ((V + U) | (V & ~M)) & full
    return m - bin(V).count("1")

def ratio(a, b):
    if not a and not b: return 100.0
    return 100.0 * 2 * lcs_len(a, b) / (len(a) + len(b))

def partial_ratio(a, b):
    if not a or not b: return 0.0
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    if len(s) == len(l): return ratio(s, l)
    best = 0.0
    for i in range(len(l) - len(s) + 1):
        best = max(best, ratio(s, l[i:i + len(s)]))
        if best >= 100: break
    return best

def token_ratio(a, b):
    return ratio(" ".join(sorted(a.split())), " ".join(sorted(b.split())))

def jaccard(a, b):
    A, B = set(a.split()), set(b.split())
    return len(A & B) / len(A | B) if (A | B) else 0.0

def canon_country(c):
    return COUNTRY_CANON.get((c or "").strip().upper(), (c or "").strip().upper())

FEATS = ["name_exact", "name_ratio", "name_partial_ratio", "name_token_ratio", "name_jaccard", "address_exact",
         "address_ratio", "address_partial_ratio", "address_jaccard", "country_match", "name_length_difference",
         "address_length_difference", "common_name_tokens", "common_address_tokens"]

def pair_features(r1, r2):
    n1, n2 = norm(r1["business_name"]), norm(r2["business_name"])
    a1, a2 = norm(r1["business_address"]), norm(r2["business_address"])
    f = dict(name_exact=int(n1 == n2), name_ratio=round(ratio(n1, n2), 2), name_partial_ratio=round(partial_ratio(n1, n2), 2),
             name_token_ratio=round(token_ratio(n1, n2), 2), name_jaccard=round(jaccard(n1, n2), 4),
             name_length_difference=abs(len(n1) - len(n2)), common_name_tokens=len(set(n1.split()) & set(n2.split())))
    if a1 and a2:
        f.update(address_exact=int(a1 == a2), address_ratio=round(ratio(a1, a2), 2),
                 address_partial_ratio=round(partial_ratio(a1, a2), 2), address_jaccard=round(jaccard(a1, a2), 4),
                 address_length_difference=abs(len(a1) - len(a2)), common_address_tokens=len(set(a1.split()) & set(a2.split())))
    else:  # missing address on either side -> blank (NaN in pandas), 0 shared tokens
        f.update(address_exact=0, address_ratio="", address_partial_ratio="", address_jaccard="",
                 address_length_difference="", common_address_tokens=0)
    c1, c2 = canon_country(r1["country"]), canon_country(r2["country"])
    f["country_match"] = -1 if (not c1 or not c2) else int(c1 == c2)
    return f

STOP = {"and", "the", "of", "pvt", "ltd", "limited", "private", "inc", "llc", "co", "company", "corp", "corporation",
        "llp", "plc", "pty", "pte", "gmbh", "ag", "sarl", "sas", "sa"}

def build_pairs(s1, s2, s3, truth, seed=42):
    rng = random.Random(seed)
    cands = [dict(r, source="S2") for r in s2] + [dict(r, source="S3") for r in s3]
    idx_n, idx_a = defaultdict(list), defaultdict(list)
    for i, r in enumerate(cands):
        for t in set(norm(r["business_name"]).split()) - STOP: idx_n[t].append(i)
        for t in set(x for x in norm(r["business_address"]).split() if x.isalpha() and len(x) >= 4): idx_a[t].append(i)
    tie = [rng.random() for _ in cands]
    pairs = set(); rows = []
    for e in s1:
        true = {i for i, r in enumerate(cands) if truth.get(r["entity_id"]) == e["entity_id"]} if False else None
    pos_by_s1 = defaultdict(set)
    for i, r in enumerate(cands):
        if r["entity_id"] in truth: pos_by_s1[truth[r["entity_id"]]].add(i)
    for e in s1:
        sc = Counter()
        for t in set(norm(e["business_name"]).split()) - STOP:
            if len(idx_n[t]) <= 200:
                for i in idx_n[t]: sc[i] += 3
        for t in set(x for x in norm(e["business_address"]).split() if x.isalpha() and len(x) >= 4):
            if len(idx_a[t]) <= 350:
                for i in idx_a[t]: sc[i] += 1
        true = pos_by_s1.get(e["entity_id"], set())
        ranked = [i for i, _ in sorted(sc.items(), key=lambda kv: (-kv[1], tie[kv[0]])) if i not in true]
        chosen = ranked[:3]
        rest = ranked[3:]
        chosen += rng.sample(rest, min(2, len(rest)))
        while True:
            j = rng.randrange(len(cands))
            if j not in true and j not in chosen: chosen.append(j); break
        for i in list(true) + chosen:
            if (e["entity_id"], i) in pairs: continue
            pairs.add((e["entity_id"], i))
            c = cands[i]
            f = pair_features(e, c)
            row = dict(source1_entity_id=e["entity_id"], candidate_entity_id=c["entity_id"], candidate_source=c["source"])
            row.update(f); row["label"] = int(i in true)
            rows.append(row)
    perm = np.random.permutation(len(rows))
    return [rows[i] for i in perm]

# ----------------------------------------------------------------------------
# 8. WRITERS / VALIDATION / STATS
# ----------------------------------------------------------------------------
def write_tsv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n")
        w.writerow(header)
        for r in rows: w.writerow(r)

def validate(split, out, gt_rows):
    s1, s2, s3 = out["S1"], out["S2"], out["S3"]
    ids = {"S1": {r["entity_id"] for r in s1}, "S2": {r["entity_id"] for r in s2}, "S3": {r["entity_id"] for r in s3}}
    res = []
    def chk(name, ok): res.append((name, bool(ok)))
    chk("1 no duplicate IDs in any source", all(len(ids[k]) == len(out[k]) for k in ids))
    chk("2 all S1 ids start with S1-", all(i.startswith("S1-") for i in ids["S1"]))
    chk("3 all S2 ids start with S2-", all(i.startswith("S2-") for i in ids["S2"]))
    chk("4 all S3 ids start with S3-", all(i.startswith("S3-") for i in ids["S3"]))
    chk("5 every S1 id exactly once in ground truth", [g[0] for g in gt_rows] == [r["entity_id"] for r in s1] and len({g[0] for g in gt_rows}) == len(s1))
    allm = [m for _, ms in gt_rows for m in ms]
    chk("6 every matched id exists in S2/S3", all(m in ids["S2"] or m in ids["S3"] for m in allm))
    chk("7 no S1 id used as a match", not any(m.startswith("S1-") for m in allm))
    chk("8 no duplicate matched ids", len(allm) == len(set(allm)))
    chk("9 some S1 entities have multiple matches", sum(len(ms) > 1 for _, ms in gt_rows) > 0)
    chk("10 some S1 entities have zero matches", sum(len(ms) == 0 for _, ms in gt_rows) > 0)
    chk("11 hard negatives present", sum(r["kind"].startswith("decoy_") for r in s2 + s3) > 0 and sum(r["origin"] == "chain_branch" for r in s1) > 0)
    chk("12 spelling errors present", any("typo" in r["name_ops"] for r in s2 + s3))
    chk("13 abbreviation variations present", any("abbrev" in r["name_ops"] or "legal_variant" in r["name_ops"] for r in s2 + s3))
    chk("14 address variations present", any(r["addr_ops"] for r in s2 + s3))
    chk("15 missing fields present", any(not r["address"] or not r["country"] for r in s2 + s3))
    chk("16 country variations present", len({r["country"] for r in s2 + s3}) > len(COUNTRY_VARIANTS))
    chk("17 no external data used (all generated in-script)", True)
    # ID leakage: fraction of matched pairs where numeric suffix equals S1's numeric suffix (expect ~0)
    same = sum(int(m[3:]) == int(s[3:]) for s, ms in gt_rows for m in ms)
    chk("18 IDs do not leak matches (numeric coincidences < 1%)", same < .01 * max(1, len(allm)))
    chk("19 no empty business_name anywhere", all(r["business_name"].strip() for r in s1) and all(r["name"].strip() for r in s2 + s3))
    chk("20 no tabs/newlines inside fields", all("\t" not in (r["name"] + r["address"]) and "\n" not in (r["name"] + r["address"]) for r in s2 + s3))
    return res

def write_split(split, out, gt_rows, base):
    s1h = ["entity_id", "business_name", "business_address", "country"]
    write_tsv(f"{base}/dataset/{split}/{split}_source1.tsv", s1h, [[r["entity_id"], r["business_name"], r["business_address"], r["country"]] for r in out["S1"]])
    for src in ("S2", "S3"):
        rows = sorted(out[src], key=lambda r: r["entity_id"])
        write_tsv(f"{base}/dataset/{split}/{split}_source{src[1]}.tsv", s1h, [[r["entity_id"], r["name"], r["address"], r["country"]] for r in rows])
    gtp = f"{base}/dataset/train/train_ground_truth.tsv" if split == "train" else f"{base}/evaluation/test_ground_truth.tsv"
    write_tsv(gtp, ["source1_entity_id", "matched_entity_ids"], [[e, ",".join(ms)] for e, ms in gt_rows])
    audit = [[r["entity_id"], src, r["master"], r["kind"], r["profile"], "|".join(r["name_ops"]), "|".join(r["addr_ops"]), r["country_mode"]]
             for src in ("S2", "S3") for r in sorted(out[src], key=lambda r: r["entity_id"])]
    audit += [[r["entity_id"], "S1", r["master"], r["origin"], "", "", "", ""] for r in out["S1"]]
    write_tsv(f"{base}/evaluation/audit_{split}.tsv", ["entity_id", "source", "master_entity_id", "record_kind", "noise_profile", "name_ops", "addr_ops", "country_mode"], audit)

def normalize_records(out):
    """Give S2/S3 rows the same key names as S1 for feature extraction."""
    return ([dict(entity_id=r["entity_id"], business_name=r["business_name"], business_address=r["business_address"], country=r["country"]) for r in out["S1"]],
            [dict(entity_id=r["entity_id"], business_name=r["name"], business_address=r["address"], country=r["country"]) for r in out["S2"]],
            [dict(entity_id=r["entity_id"], business_name=r["name"], business_address=r["address"], country=r["country"]) for r in out["S3"]])

def compute_stats(out, gt_rows, pairs):
    st = {}
    st["records"] = {k: len(out[k]) for k in ("S1", "S2", "S3")}
    st["unique_underlying_businesses_S1"] = len({r["master"] for r in out["S1"]})
    st["unique_underlying_businesses_incl_decoys"] = len({r["master"] for k in ("S1", "S2", "S3") for r in out[k]})
    st["positive_matches"] = sum(len(ms) for _, ms in gt_rows)
    st["positives_S2"] = sum(1 for _, ms in gt_rows for m in ms if m.startswith("S2"))
    st["positives_S3"] = sum(1 for _, ms in gt_rows for m in ms if m.startswith("S3"))
    st["singleton_s1"] = sum(len(ms) == 0 for _, ms in gt_rows)
    st["single_match_s1"] = sum(len(ms) == 1 for _, ms in gt_rows)
    st["multi_match_s1"] = sum(len(ms) > 1 for _, ms in gt_rows)
    st["max_matches_one_s1"] = max(len(ms) for _, ms in gt_rows)
    st["cross_source_multi"] = sum(any(m.startswith("S2") for m in ms) and any(m.startswith("S3") for m in ms) for _, ms in gt_rows)
    st["orphan_records"] = {k: sum(r["kind"].startswith("decoy_") for r in out[k]) for k in ("S2", "S3")}
    st["decoy_kinds"] = dict(Counter(r["kind"] for k in ("S2", "S3") for r in out[k] if r["kind"].startswith("decoy_")))
    st["s1_origins"] = dict(Counter(r["origin"] for r in out["S1"]))
    st["s1_countries"] = dict(Counter(r["country"] for r in out["S1"]))
    miss = {}
    for k in ("S2", "S3"):
        n = len(out[k])
        miss[k] = dict(address_missing_pct=round(100 * sum(not r["address"] for r in out[k]) / n, 2),
                       country_missing_pct=round(100 * sum(not r["country"] for r in out[k]) / n, 2),
                       pin_missing_pct=round(100 * sum(r["flags"].get("missing_addr") or r["flags"].get("pin_missing") for r in out[k]) / n, 2))
    miss["S1"] = dict(address_missing_pct=0.0, country_missing_pct=0.0, pin_missing_pct=0.0)
    st["missing"] = miss
    st["noise_profile"] = {k: {p: round(100 * c / len(out[k]), 1) for p, c in Counter(r["profile"] for r in out[k]).most_common()} for k in ("S2", "S3")}
    allr = out["S2"] + out["S3"]
    st["noise_profile_overall"] = {p: round(100 * c / len(allr), 1) for p, c in Counter(r["profile"] for r in allr).most_common()}
    st["name_op_freq_pct"] = {k: round(100 * c / len(out[k]), 1) for k in ("S2", "S3") for c in [0]}
    st["name_op_freq_pct"] = {k: {o: round(100 * c / len(out[k]), 1) for o, c in Counter(o for r in out[k] for o in set(r["name_ops"])).most_common()} for k in ("S2", "S3")}
    st["addr_op_freq_pct"] = {k: {o: round(100 * c / len(out[k]), 1) for o, c in Counter(o for r in out[k] for o in set(r["addr_ops"])).most_common()} for k in ("S2", "S3")}
    st["country_variant_pct"] = {k: round(100 * sum(r["country_mode"] == "variant" for r in out[k]) / len(out[k]), 1) for k in ("S2", "S3")}
    if pairs is not None:
        lab = [p["label"] for p in pairs]
        st["pairs_total"] = len(pairs); st["pairs_pos"] = sum(lab); st["pairs_neg"] = len(lab) - sum(lab)
        def hard(p):
            return p["label"] == 0 and (p["name_ratio"] >= 80 or (p["address_ratio"] != "" and p["address_ratio"] >= 85))
        st["pairs_hard_neg"] = sum(hard(p) for p in pairs)
        st["pairs_same_name_diff_biz"] = sum(p["label"] == 0 and p["name_exact"] == 1 for p in pairs)
        st["pairs_same_addr_diff_biz"] = sum(p["label"] == 0 and p["address_exact"] == 1 for p in pairs)
        st["pairs_by_source"] = dict(Counter(p["candidate_source"] for p in pairs))
    return st

def baselines(out, gt_rows):
    s1, s2, s3 = normalize_records(out)
    truth = {m: e for e, ms in gt_rows for m in ms}
    tp_all = len(truth)
    cands = s2 + s3
    res = {}
    for name, keyf in (("exact_name", lambda r: norm(r["business_name"])),
                       ("exact_name_and_address", lambda r: (norm(r["business_name"]), norm(r["business_address"])))):
        d = defaultdict(list)
        for c in cands: d[keyf(c)].append(c["entity_id"])
        pred = tp = 0
        for e in s1:
            k = keyf(e)
            if isinstance(k, tuple) and not k[1]: continue
            for cid in d.get(k, []):
                pred += 1; tp += int(truth.get(cid) == e["entity_id"])
        res[name] = dict(predicted=pred, true_positive=tp, precision=round(tp / pred, 3) if pred else 0, recall=round(tp / tp_all, 3))
    return res

def examples(out, gt_rows):
    s1 = {r["entity_id"]: r for r in out["S1"]}
    rec = {r["entity_id"]: r for k in ("S2", "S3") for r in out[k]}
    def line(r, nm="business_name", ad="business_address"):
        return f"{r['entity_id']} | {r[nm]} | {r[ad]} | {r['country'] or '<missing>'}"
    ex = {"multi": [], "single": [], "singleton": [], "chain": [], "similar_name": [], "same_name": []}
    for e, ms in gt_rows:
        if len(ms) >= 4 and len(ex["multi"]) < 3:
            ex["multi"].append([line(s1[e])] + ["   -> " + line(rec[m], "name", "address") for m in ms])
        if len(ms) == 0 and len(ex["singleton"]) < 3:
            ex["singleton"].append(line(s1[e]) + "   -> (no matches)")
    for r in out["S1"]:
        if r["origin"] == "chain_branch" and len(ex["chain"]) < 2:
            twins = [x for x in out["S1"] if x["business_name"] == r["business_name"] and x["entity_id"] != r["entity_id"]]
            if twins: ex["chain"].append([line(r), "   vs " + line(twins[0]) + "   [DIFFERENT businesses: same name, different address]"])
    for k in ("S2", "S3"):
        for r in out[k]:
            if r["kind"] == "decoy_same_name_diff_addr" and len(ex["same_name"]) < 3:
                base = next((s for s in out["S1"] if s["master"] == r["master"]), None)
            if r["kind"] in ("decoy_same_name_diff_addr", "decoy_similar_name_diff_addr", "decoy_similar_addr_diff_name"):
                key = {"decoy_same_name_diff_addr": "same_name", "decoy_similar_name_diff_addr": "similar_name",
                       "decoy_similar_addr_diff_name": "chain"}[r["kind"]]
                if len(ex[key]) < 3 and key != "chain" or (key == "chain" and len(ex["chain"]) < 4):
                    ex[key].append(line(r, "name", "address") + "   [NOT a match for any S1 record]")
    return ex

def write_readme(base, st_tr, st_te, bl, ex, val_tr, val_te):
    def d(x): return json.dumps(x, indent=1, ensure_ascii=False)
    md = f"""# Synthetic Business Entity-Resolution Dataset

Everything in this folder is **synthetically generated** by `generate_dataset.py` (seeded with `random.seed(42)` and
`numpy.random.seed(42)`, so every run reproduces the same files byte-for-byte). No real business database, geocoder or
external file is used. Names, streets, postal codes and landmarks are invented; city/state names are generic public
geography used only as text.

## 1. Purpose
Develop and test an ML pipeline (blocking → feature engineering → classifier → threshold tuning) that decides whether
records from different sources describe the **same real-world business**, before running it on the official hackathon data.

## 2. Layout
```
dataset/train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/   test_source1.tsv   test_source2.tsv   test_source3.tsv        (no labels)
evaluation/     test_ground_truth.tsv   (PRIVATE - local scoring only)
                audit_train.tsv / audit_test.tsv   (PRIVATE - hidden master_entity_id + noise tags for error analysis)
ml/             training_pairs.tsv
generate_dataset.py
```
All files are **tab-separated, UTF-8** (addresses contain commas). Columns of every source file:
`entity_id, business_name, business_address, country`.

## 3. Sources
* **Source 1** - clean, deduplicated reference ({st_tr['records']['S1']} businesses in train). Full legal suffixes, canonical
  address `12 MG Road, Chennai, Tamil Nadu 600001`, full country name, no missing fields. 16 business categories, 8 countries
  (India/US dominant; Germany, France, Singapore rare). It deliberately contains **chain branches** (same name, different address)
  and **look-alike siblings** (similar name in the same city, or same street with a different business) - these are separate entities.
* **Source 2** - noisy; skewed toward *abbreviations, legal-suffix changes, typos/misspellings, punctuation/case*.
* **Source 3** - noisy in a *different* way; skewed toward *missing address components, landmark-based addresses, address
  re-ordering, shortened names, transliteration, missing PIN codes*.
* Both S2 and S3 also contain **orphan records** (businesses that do not exist in Source 1) that are built to be confusable
  with Source 1 businesses (see section 7).
* IDs (`S1-00001`, `S2-00001`, ...) are assigned after a random shuffle, so numbering and row order carry no information.

## 4. Ground truth
`train_ground_truth.tsv` has `source1_entity_id, matched_entity_ids` - one row per S1 id, matches comma-separated (S2 ids and S3 ids only),
empty for singletons. Internally each business has a hidden `master_entity_id`; ground truth is derived from it, so it is
consistent by construction. The hidden id never appears in the source files (only in `evaluation/audit_*.tsv`).

## 5. Noise generation
Each S2/S3 record draws one *noise profile* (weights differ per source and differ again in the test split):
`exact/near-exact, abbreviation, spelling/typo, address variation, missing field, word-order/punctuation, complex (3-4 stacked ops)`.
Name ops: legal-suffix variants (Private Limited/Pvt Ltd/Pvt. Ltd.), Corp/Co/Inc/&/and, tech/intl/engg-style abbreviations,
initials (A.B.C), dropped/shortened words, keyboard typos and common misspellings, transliteration (Lakshmi/Laxmi, v/w, ee/i), accent stripping,
word re-ordering, punctuation/case/spacing. Address ops: Rd/St/Ave/Apt/Bldg abbreviations, `No. 12` / `#12` / `12-A`, dotted `M.G.`,
missing PIN/state/city/house-number/unit, landmark replacement (`Near Gandhi Mall`), component re-ordering, alternative separators, PIN formats, typos.
Country: canonical name, variants (IN, USA, U.K., Deutschland ...) or missing. Some records get several problems at once.
**The test split uses different profile weights, different countries mix and extra unseen ops** (`The` prefix, vowel-dropped words, `Group` suffix,
umlaut expansion, `PIN 600001`/`ZIP` prefixes, unit-first addresses, country appended to the address).

## 6. Positive / singleton / multi-match examples (train)
```
""" + "\n\n".join("\n".join(e) for e in ex["multi"]) + "\n" + "\n".join(ex["singleton"]) + f"""
```

## 7. Negative examples (train)
Hard negatives are of five kinds: same name/different address, similar name/different address, same city/different business,
similar address/different name, and same country/unrelated. They come from (a) S1 chain branches and siblings, and (b) orphan decoy
records in S2/S3. Records that share a name but belong to different businesses:
```
""" + "\n".join("\n".join(e) if isinstance(e, list) else e for e in ex["chain"]) + "\n" + "\n".join(ex["same_name"]) + "\n" + "\n".join(ex["similar_name"]) + f"""
```

## 8. Load with pandas
```python
import pandas as pd
kw = dict(sep="\\t", dtype=str, keep_default_na=False, encoding="utf-8")   # keep_default_na=False keeps '' for missing fields
s1 = pd.read_csv("dataset/train/train_source1.tsv", **kw)
s2 = pd.read_csv("dataset/train/train_source2.tsv", **kw)
s3 = pd.read_csv("dataset/train/train_source3.tsv", **kw)
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", **kw)
gt["matched"] = gt.matched_entity_ids.apply(lambda x: x.split(",") if x else [])
pairs = pd.read_csv("ml/training_pairs.tsv", sep="\\t")   # blank features become NaN
```

## 9. `training_pairs.tsv`
Columns: `source1_entity_id, candidate_entity_id, candidate_source` + 14 similarity features + `label` (1 = same business).
* **Positives:** every true (S1, S2/S3) pair from the ground truth.
* **Negatives, per S1 entity:** a cheap *blocking* step (inverted index on rare name tokens, weight 3, and rare address tokens, weight 1) scores all
  S2/S3 records; the 3 highest-scoring non-matches are taken as hard negatives, 2 random non-matches from the remaining block as medium negatives,
  and 1 random record from the whole pool as an easy negative. Blocking uses record text only - **never the label**.
* **Features** are computed only from the two records' text (normalised: lower-case, accent-stripped, `&`->`and`, punctuation removed) with a pure-Python
  LCS (indel) ratio: `*_ratio`, `*_partial_ratio` (best window), `name_token_ratio` (token-sort), Jaccard, exact flags, length differences (characters),
  shared-token counts. Scores are 0-100. `country_match` = 1 same canonical country, 0 different, **-1 unknown** (either side missing).
  If either address is missing, the address similarity features are blank (NaN) and `address_exact=0`.
* Label is looked up from the ground truth only *after* features are computed.

## 10. Statistics (train)
```
{d(st_tr)}
```
Exact-string baselines on the full S1 x (S2 u S3) space (train):
```
{d(bl)}
```
Test split (600 S1 / ~900 / ~900) statistics:
```
{d({k: st_te[k] for k in ('records','positive_matches','singleton_s1','multi_match_s1','noise_profile_overall','decoy_kinds','s1_countries')})}
```
Validation checks (all must be PASS): train {sum(ok for _, ok in val_tr)}/{len(val_tr)}, test {sum(ok for _, ok in val_te)}/{len(val_te)}.

## 11. Suggested evaluation
Split **by S1 entity** (GroupKFold on `source1_entity_id`), never by pair, so near-duplicate records of one business cannot leak across folds.
Judge the whole pipeline, not only the classifier: blocking recall, pair-level precision/recall/F1 and PR-AUC, then the final
one-to-many assignment. Tune the threshold on out-of-fold predictions; keep the test split untouched until the end and score with
`evaluation/test_ground_truth.tsv`. Report per-source (S2 vs S3), per-country and per-noise-profile (via `audit_*.tsv`) metrics.
"""
    with open(f"{base}/README.md", "w", encoding="utf-8") as fh:
        fh.write(md)

def main():
    base = ROOT
    results = {}
    for split in ("train", "test"):
        out, gt_rows, mid2s1 = gen_split(split)
        write_split(split, out, gt_rows, base)
        results[split] = (out, gt_rows)
    out_tr, gt_tr = results["train"]; out_te, gt_te = results["test"]
    s1, s2, s3 = normalize_records(out_tr)
    truth = {m: e for e, ms in gt_tr for m in ms}
    pairs = build_pairs(s1, s2, s3, truth)
    header = ["source1_entity_id", "candidate_entity_id", "candidate_source"] + FEATS + ["label"]
    write_tsv(f"{base}/ml/training_pairs.tsv", header, [[p[h] for h in header] for p in pairs])
    st_tr = compute_stats(out_tr, gt_tr, pairs)
    st_te = compute_stats(out_te, gt_te, None)
    bl = baselines(out_tr, gt_tr)
    val_tr = validate("train", out_tr, gt_tr); val_te = validate("test", out_te, gt_te)
    ex = examples(out_tr, gt_tr)
    write_readme(base, st_tr, st_te, bl, ex, val_tr, val_te)
    with open(f"{base}/evaluation/stats_train.json", "w") as fh: json.dump(dict(train=st_tr, test=st_te, baselines=bl), fh, indent=1)
    print("VALIDATION (train / test):")
    for (n, a), (_, b) in zip(val_tr, val_te): print(f"  [{'PASS' if a else 'FAIL'}/{'PASS' if b else 'FAIL'}] {n}")
    print(json.dumps(st_tr, indent=1)); print("BASELINES", bl)
    assert all(ok for _, ok in val_tr + val_te), "validation failed"

if __name__ == "__main__":
    main()
