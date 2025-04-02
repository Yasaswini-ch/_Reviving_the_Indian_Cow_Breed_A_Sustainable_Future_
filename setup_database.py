import sqlite3
import pandas as pd
import datetime # Import datetime for timestamps if needed manually (though DEFAULT works)

# Connect to the database
db_file = 'Cows.db'
print(f"Connecting to database: {db_file}")
connection = sqlite3.connect(db_file)
cursor = connection.cursor()

# --- Create Tables (with UNIQUE constraints where appropriate) ---
print("Creating tables if they don't exist...")

cursor.execute('''
CREATE TABLE IF NOT EXISTS cattle_breeds (
    breed_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE, -- Added UNIQUE constraint
    region TEXT,
    milk_yield INTEGER,
    strength TEXT,
    lifespan INTEGER,
    image_url TEXT -- Changed from image to image_url for clarity
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS government_schemes (
    scheme_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    details TEXT NOT NULL,
    region TEXT, -- Removed NOT NULL constraint to allow Central schemes easily
    type TEXT, -- Added Scheme Type column
    url TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS breeding_pairs (
    pair_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cattle_1 TEXT,      -- ID or Name of the first cattle
    cattle_2 TEXT,      -- ID or Name of the second cattle
    goal TEXT,          -- Breeding goal (e.g., 'High Milk Yield', 'Disease Resistance')
    genetic_score INTEGER, -- Added placeholder for compatibility score
    status TEXT,        -- Added status (e.g., 'Suggested', 'Confirmed', 'Completed')
    notes TEXT,         -- Added notes field
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS offspring_data (
    offspring_data_id INTEGER PRIMARY KEY AUTOINCREMENT, -- Renamed primary key for clarity
    parent_1 TEXT,      -- ID or Name of the first parent
    parent_2 TEXT,      -- ID or Name of the second parent
    offspring_id TEXT UNIQUE, -- Added specific ID for the offspring, made unique
    breed TEXT,
    sex TEXT,           -- Added Sex (Male/Female)
    dob DATE,           -- Added Date of Birth
    predicted_traits TEXT, -- Kept this field, could store JSON or comma-separated values
    actual_traits TEXT, -- Added field for observed traits later
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS eco_practices (
    practice_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE, -- Added UNIQUE constraint
    description TEXT, -- Added description
    category TEXT, -- Added category (e.g., 'Manure Management', 'Water Conservation')
    suitability TEXT, -- Added suitability (e.g., 'Cattle Farms', 'Crop Farms', 'Both')
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    -- Removed efficiency/co2 saved as they are better calculated dynamically
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS image_analysis (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT, -- Changed from url to path, assuming local storage or identifier
    uploaded_filename TEXT, -- Added original filename
    detected_breed TEXT,
    confidence_score FLOAT, -- Added confidence score
    analysis_backend TEXT, -- Added which backend did the analysis (e.g., 'YOLOv8', 'API')
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS user_queries (
    query_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT, -- Added session ID for multi-user context
    user_input TEXT,
    user_language TEXT, -- Renamed from 'language'
    translated_input TEXT, -- Added field for English translation if needed
    bot_response TEXT,
    response_language TEXT, -- Added field for the language of the response
    model_used TEXT, -- Added which model generated response (e.g., 'Gemini-1.5')
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS price_trends (
    trend_id INTEGER PRIMARY KEY AUTOINCREMENT,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL, -- Added month for more granularity
    breed TEXT, -- Added breed specificity
    region TEXT, -- Added region specificity
    average_price FLOAT,
    UNIQUE(year, month, breed, region) -- Composite UNIQUE key
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS disease_diagnosis (
    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
    symptoms TEXT NOT NULL,
    detected_disease TEXT, -- Removed UNIQUE constraint, multiple entries might mention same disease with different symptoms/context
    suggested_treatment TEXT,
    severity TEXT, -- Added severity (e.g., 'Low', 'Medium', 'High')
    notes TEXT, -- Added general notes field
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')
print("Tables created/verified successfully.")
connection.commit() # Commit table creations


# --- Insert Sample Data ---

# Government Schemes (URLs are NOT modified)
print("\n--- Processing Government Schemes ---")
def insert_government_schemes(cursor):
    schemes = [
        ('Rashtriya Gokul Mission', 'Promotes indigenous cattle breeding and genetic improvement.', 'All India / Central', 'Animal Husbandry', 'https://dahd.nic.in/schemes/programmes/rashtriya-gokul-mission'),
        ('National Livestock Mission (NLM)', 'Sustainable development of livestock sector, covering feed/fodder, breed improvement, entrepreneurship.', 'All India / Central', 'Animal Husbandry', 'https://dahd.nic.in/nlm'),
        ('Dairy Entrepreneurship Development Scheme (DEDS - replaced by DIDF aspects)', 'Financial support for setting up small dairy farms & units (Check NABARD/NDDB for current alternatives like DIDF).', 'All India / Central', 'Dairy Development', 'https://www.nabard.org/content1.aspx?id=591'),
        ('Kisan Credit Card (KCC) Scheme', 'Provides short-term credit to farmers for agriculture and allied activities including animal husbandry.', 'All India / Central', 'Finance/Credit', 'https://pmkisan.gov.in/kcc/'),
        ('PM-KUSUM', 'Promotes solar energy use in agriculture, including solar pumps and potentially solar power for dairy farms/biogas plants.', 'All India / Central', 'Energy/Agriculture', 'https://pmkusum.mnre.gov.in/'),
        ('National Programme for Dairy Development (NPDD)', 'Aims to strengthen dairy cooperatives and increase milk production.', 'All India / Central', 'Dairy Development', 'https://dahd.nic.in/npdd'),
        ('Livestock Health & Disease Control (LH&DC) Programme', 'Focuses on prevention, control and eradication of animal diseases, including FMD, Brucellosis.', 'All India / Central', 'Animal Health', 'https://dahd.nic.in/lh-dc'),
        ('Animal Husbandry Infrastructure Development Fund (AHIDF)', 'Incentivizes investments in dairy processing, value addition infrastructure, and animal feed plants.', 'All India / Central', 'Infrastructure', 'https://ahidf.udyamimitra.in/'),
        ('Gobar Dhan Scheme', 'Promotes converting cattle dung and solid waste into compost, biogas, and biofuel.', 'All India / Central', 'Waste Management/Energy', 'https://sbm.gov.in/Gobardhan/'),
        # --- State Specific Examples (Illustrative - URLs are NOT modified) ---
        ('Mukhyamantri Dugdh Utpadak Sambal Yojana (Rajasthan)', 'Provides bonus per litre of milk to cooperative dairy farmers.', 'Rajasthan', 'Subsidy/Incentive', 'https://animalhusbandry.rajasthan.gov.in/'), # URL might be generic
        ('Mukhyamantri Gau Mata Poshan Yojana (Gujarat)', 'Assistance for maintenance of unproductive/old cattle in Gaushalas/Panjrapoles.', 'Gujarat', 'Animal Welfare', 'https://cmogujarat.gov.in/en/latest-news/greeting-ceremony-cm-announcing-mukhyamantri-gaumata-poshan-yojana'),
        ('Ksheera Santhwanam (Kerala)', 'Insurance scheme for dairy farmers covering cattle death.', 'Kerala', 'Insurance/Welfare', 'https://ksheerasree.kerala.gov.in/'),
        ('Nand Baba Milk Mission (Uttar Pradesh)', 'Aims to enhance dairy infrastructure and market access for milk producers.', 'Uttar Pradesh', 'Dairy Development', 'https://updairydevelopment.gov.in/'),
        # Add more central/state schemes if needed
    ]

    inserted_count = 0
    skipped_count = 0
    for name, details, region, type, url in schemes:
        try:
            cursor.execute('''
                INSERT INTO government_schemes (name, details, region, type, url)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, details, region, type, url))
            if cursor.rowcount > 0:
                inserted_count += 1
            else:
                 skipped_count += 1
        except sqlite3.IntegrityError:
            skipped_count += 1
        except Exception as e:
            print(f"Error inserting scheme {name}: {e}")
            skipped_count += 1

    print(f"Government Schemes: Inserted {inserted_count}, Skipped {skipped_count} duplicates.")
    connection.commit()

insert_government_schemes(cursor) # Call the function


# Cattle Breeds Data
print("\n--- Processing Cattle Breeds ---")
cattle_breeds_data = [
    ("Gir", "Gujarat", 12, "High", 18, "images/gir.jpg"),
    ("Sahiwal", "Punjab", 14, "Medium", 20, "images/sahiwal.jpg"),
    ("Ongole", "Andhra Pradesh", 10, "Very High", 22, "images/ongole.jpg"),
    ("Punganur", "Andhra Pradesh", 6, "Low", 15, "images/punganur.jpg"),
    ("Amrit Mahal", "Karnataka", 7, "High", 18, "images/amrit_mahal.jpg"),
    ("Deoni", "Maharashtra", 9, "Medium", 19, "images/deoni.jpg"),
    ("Hallikar", "Karnataka", 8, "Very High", 20, "images/hallikar.jpg"),
    ("Kankrej", "Gujarat", 11, "High", 21, "images/kankrej.jpg"),
    ("Krishna Valley", "Karnataka", 7, "Very High", 19, "images/krishna_valley.jpg"),
    ("Malnad Gidda", "Karnataka", 5, "Medium", 16, "images/malnad_gidda.jpg"),
    ("Rathi", "Rajasthan", 10, "Medium", 20, "images/rathi.jpg"),
    ("Red Sindhi", "Sindh (Origin)", 11, "High", 22, "images/red_sindhi.jpg"),
    ("Tharparkar", "Rajasthan", 9, "Medium", 21, "images/tharparkar.jpg")
]
inserted_count_breeds = 0
skipped_count_breeds = 0
for breed in cattle_breeds_data:
    try:
        cursor.execute("INSERT INTO cattle_breeds (name, region, milk_yield, strength, lifespan, image_url) VALUES (?, ?, ?, ?, ?, ?)", breed)
        if cursor.rowcount > 0:
            inserted_count_breeds += 1
        else:
             skipped_count_breeds += 1
    except sqlite3.IntegrityError:
        skipped_count_breeds += 1
    except Exception as e:
        print(f"Error inserting breed {breed[0]}: {e}")
        skipped_count_breeds += 1
print(f"Cattle Breeds: Inserted {inserted_count_breeds}, Skipped {skipped_count_breeds} duplicates.")
connection.commit()

# Breeding Pairs Data
print("\n--- Processing Breeding Pairs ---")
breeding_pairs_data = [
    ('GIR-BULL-01', 'GIR-COW-A5', 'High Milk Yield', 85, 'Suggested', 'Good match for milk traits.', datetime.datetime.now() - datetime.timedelta(days=10)),
    ('SAH-BULL-03', 'SAH-COW-B2', 'Breed Purity', 92, 'Suggested', 'Excellent lineage match.', datetime.datetime.now() - datetime.timedelta(days=8)),
    ('ONG-BULL-X1', 'KANK-COW-C7', 'Dual Purpose (Milk & Draft)', 78, 'Suggested', 'Potential hybrid vigor for strength and moderate milk.', datetime.datetime.now() - datetime.timedelta(days=5)),
    ('RATHI-BULL-R2', 'RATHI-COW-D1', 'Drought Tolerance', 88, 'Completed', 'Successful pairing, offspring recorded.', datetime.datetime.now() - datetime.timedelta(days=30)),
    ('GIR-BULL-01', 'GIR-COW-A8', 'High Milk Yield', 82, 'Suggested', 'Alternative pairing for milk.', datetime.datetime.now() - datetime.timedelta(days=2)),
    ('HALLIKAR-BULL-H1', 'AMRIT-COW-AM2', 'High Draft Power', 90, 'Confirmed', 'Scheduled for AI next cycle.', datetime.datetime.now() - datetime.timedelta(days=1)),
]
inserted_count_pairs = 0
for pair in breeding_pairs_data:
     try:
        cursor.execute('''INSERT INTO breeding_pairs
                          (cattle_1, cattle_2, goal, genetic_score, status, notes, timestamp)
                          VALUES (?, ?, ?, ?, ?, ?, ?)''', pair)
        inserted_count_pairs += 1
     except Exception as e:
        print(f"Error inserting breeding pair {pair[0]}/{pair[1]}: {e}")

print(f"Breeding Pairs: Inserted {inserted_count_pairs} records.")
connection.commit()

# Offspring Data
print("\n--- Processing Offspring Data ---")
offspring_data_list = [
    ('RATHI-BULL-R2', 'RATHI-COW-D1', 'RATHI-CALF-001', 'Rathi', 'Female', '2023-11-15', 'Good confirmation, expected moderate milk', 'Developing well, good temperament', datetime.datetime.now() - datetime.timedelta(days=25)),
    ('GIR-BULL-01', 'GIR-COW-A5', 'GIR-CALF-001', 'Gir', 'Male', '2024-01-20', 'High milk potential (dam side), good frame', None, datetime.datetime.now() - datetime.timedelta(days=5)),
    ('SAH-BULL-03', 'SAH-COW-B2', 'SAH-CALF-001', 'Sahiwal', 'Female', '2024-02-10', 'Excellent breed characteristics, high milk potential', None, datetime.datetime.now() - datetime.timedelta(days=1)),
]
inserted_count_offspring = 0
skipped_count_offspring = 0
for offspring in offspring_data_list:
    try:
        cursor.execute('''INSERT INTO offspring_data
                          (parent_1, parent_2, offspring_id, breed, sex, dob, predicted_traits, actual_traits, timestamp)
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', offspring)
        if cursor.rowcount > 0:
             inserted_count_offspring += 1
        else:
             skipped_count_offspring += 1
    except sqlite3.IntegrityError:
         skipped_count_offspring += 1
    except Exception as e:
        print(f"Error inserting offspring {offspring[2]}: {e}")
        skipped_count_offspring += 1
print(f"Offspring Data: Inserted {inserted_count_offspring}, Skipped {skipped_count_offspring} duplicates (based on offspring_id).")
connection.commit()

# Eco Practices Data
print("\n--- Processing Eco Practices ---")
eco_practices_data = [
    ('Manure Composting', 'Decomposing manure with crop residues to create rich organic fertilizer.', 'Manure Management', 'Cattle Farms'),
    ('Biogas Production', 'Anaerobic digestion of dung to produce cooking gas and slurry.', 'Manure Management/Energy', 'Cattle Farms'),
    ('Rainwater Harvesting', 'Collecting and storing rainwater for livestock drinking or irrigation.', 'Water Conservation', 'Both'),
    ('Drip Irrigation (for Fodder)', 'Efficient water delivery directly to fodder crop roots.', 'Water Conservation', 'Crop Farms/Both'),
    ('Rotational Grazing', 'Moving livestock between paddocks to improve pasture health.', 'Pasture Management', 'Cattle Farms'),
    ('Silvopasture', 'Integrating trees with pasture for fodder, shade, and biodiversity.', 'Agroforestry', 'Cattle Farms'),
    ('Vermicomposting', 'Using earthworms to convert dung/organic waste into high-quality compost.', 'Manure Management', 'Both'),
    ('Integrated Pest Management (IPM)', 'Using biological and cultural methods to control pests in fodder crops.', 'Crop Management', 'Crop Farms/Both'),
]
inserted_count_eco = 0
skipped_count_eco = 0
for practice in eco_practices_data:
    try:
        cursor.execute("INSERT INTO eco_practices (name, description, category, suitability) VALUES (?, ?, ?, ?)", practice)
        if cursor.rowcount > 0:
            inserted_count_eco += 1
        else:
            skipped_count_eco += 1
    except sqlite3.IntegrityError:
        skipped_count_eco += 1
    except Exception as e:
        print(f"Error inserting eco practice {practice[0]}: {e}")
        skipped_count_eco += 1
print(f"Eco Practices: Inserted {inserted_count_eco}, Skipped {skipped_count_eco} duplicates.")
connection.commit()


# Image Analysis Data
print("\n--- Processing Image Analysis ---")
image_analysis_data = [
    ('uploads/gir_cow_1.jpg', 'gir_cow_1.jpg', 'Gir', 0.92, 'YOLOv8-Custom', datetime.datetime.now() - datetime.timedelta(hours=5)),
    ('uploads/sahiwal_side.png', 'sahiwal_side.png', 'Sahiwal', 0.88, 'YOLOv8-Custom', datetime.datetime.now() - datetime.timedelta(hours=4)),
    ('uploads/group_cattle.jpeg', 'group_cattle.jpeg', 'Multiple/Unclear', 0.45, 'YOLOv8-Custom', datetime.datetime.now() - datetime.timedelta(hours=3)),
    ('user_images/unknown_calf.jpg', 'unknown_calf.jpg', 'Undetermined', 0.30, 'YOLOv8-Custom', datetime.datetime.now() - datetime.timedelta(hours=2)),
    ('api_uploads/img_012.jpg', 'img_012.jpg', 'Kankrej', 0.75, 'ExternalAPI-XYZ', datetime.datetime.now() - datetime.timedelta(hours=1)),
]
inserted_count_img = 0
for img_data in image_analysis_data:
    try:
        cursor.execute('''INSERT INTO image_analysis
                        (image_path, uploaded_filename, detected_breed, confidence_score, analysis_backend, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)''', img_data)
        inserted_count_img += 1
    except Exception as e:
        print(f"Error inserting image analysis for {img_data[0]}: {e}")
print(f"Image Analysis: Inserted {inserted_count_img} records.")
connection.commit()

# User Queries Data
print("\n--- Processing User Queries ---")
user_queries_data = [
    ('session_abc', 'Tell me about Gir cows', 'en', None, 'Gir cows originate from Gujarat, known for high milk yield and heat tolerance. They have distinctive curved horns.', 'en', 'Gemini-1.5', datetime.datetime.now() - datetime.timedelta(minutes=30)),
    ('session_xyz', 'साहीवाल गाय की जानकारी दें', 'hi', 'Give information about Sahiwal cows', 'साहीवाल गाय पंजाब क्षेत्र की एक प्रमुख डेयरी नस्ल है। वे अपनी उच्च दूध उत्पादन क्षमता और गर्मी सहनशीलता के लिए जानी जाती हैं।', 'hi', 'Gemini-1.5', datetime.datetime.now() - datetime.timedelta(minutes=25)),
    ('session_pqr', 'How to compost cow dung?', 'en', None, 'To compost cow dung, mix it with brown materials like dry leaves or straw. Keep the pile moist and turn it regularly for aeration. It takes several weeks to months.', 'en', 'Gemini-1.5', datetime.datetime.now() - datetime.timedelta(minutes=20)),
    ('session_abc', 'What is FMD?', 'en', None, 'FMD (Foot-and-Mouth Disease) is a highly contagious viral disease affecting cattle. Symptoms include blisters, fever, and lameness. Vaccination is key for prevention. Consult a vet immediately if suspected.', 'en', 'Gemini-1.5', datetime.datetime.now() - datetime.timedelta(minutes=15)),
    ('session_lmn', 'Biogas plant subsidy?', 'en', 'Biogas plant subsidy?', 'Government schemes like Gobar Dhan and support from MNRE often provide subsidies for biogas plants using cow dung. Check official portals like MNRE or state nodal agencies for details.', 'en', 'Gemini-1.5', datetime.datetime.now() - datetime.timedelta(minutes=10)),
]
inserted_count_queries = 0
for query_data in user_queries_data:
     try:
        cursor.execute('''INSERT INTO user_queries
                        (session_id, user_input, user_language, translated_input, bot_response, response_language, model_used, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', query_data)
        inserted_count_queries += 1
     except Exception as e:
        print(f"Error inserting user query '{query_data[1]}': {e}")
print(f"User Queries: Inserted {inserted_count_queries} records.")
connection.commit()


# Price Trends Data
print("\n--- Processing Price Trends ---")
price_data = [
    # Year, Month, Breed, Region, Avg_Price
    (2023, 10, 'Gir', 'Gujarat', 65000),
    (2023, 10, 'Sahiwal', 'Punjab', 68000),
    (2023, 11, 'Gir', 'Gujarat', 66000),
    (2023, 11, 'Sahiwal', 'Punjab', 67500),
    (2023, 12, 'Gir', 'Gujarat', 67000),
    (2023, 12, 'Sahiwal', 'Punjab', 69000),
    (2023, 12, 'Crossbred', 'Maharashtra', 55000),
    (2024, 1, 'Gir', 'Gujarat', 68000),
    (2024, 1, 'Sahiwal', 'Punjab', 70000),
    (2024, 1, 'Crossbred', 'Maharashtra', 56000),
    (2024, 2, 'Gir', 'Gujarat', 68500),
    (2024, 2, 'Sahiwal', 'Punjab', 71000),
    (2024, 2, 'Ongole', 'Andhra Pradesh', 60000),
]

inserted_count_price = 0
skipped_count_price = 0
for trend in price_data:
    try:
        cursor.execute("INSERT INTO price_trends (year, month, breed, region, average_price) VALUES (?, ?, ?, ?, ?)", trend)
        if cursor.rowcount > 0:
            inserted_count_price += 1
        else:
             skipped_count_price += 1
    except sqlite3.IntegrityError:
        skipped_count_price += 1
    except Exception as e:
        print(f"Error inserting price trend {trend}: {e}")
        skipped_count_price += 1
print(f"Price Trends: Inserted {inserted_count_price}, Skipped {skipped_count_price} duplicates.")
connection.commit()


# Disease Diagnosis Data
print("\n--- Processing Disease Diagnosis ---")
disease_data = [
    ('High fever, shivering, nasal discharge, cough, difficulty breathing', 'Bovine Respiratory Disease (BRD) Complex', 'Consult Vet. Antibiotics (if bacterial), anti-inflammatories, supportive care (fluids, rest), improve ventilation.', 'Medium to High', 'Common in young/stressed cattle.'),
    ('Watery diarrhea (sometimes bloody), dehydration, weakness, loss of appetite (esp. calves)', 'Scours (Calf Diarrhea)', 'Consult Vet. Fluid therapy (oral/IV electrolytes) is critical. Identify cause (viral, bacterial, protozoal) for specific treatment. Keep calf warm & clean.', 'High (in calves)', 'Multiple causes. Hygiene is key prevention.'),
    ('Sudden high fever, lameness, swelling with gas/crackling sound in large muscles (hip, shoulder)', 'Black Quarter (BQ)', 'Consult Vet Immediately. High dose Penicillin if caught extremely early. Often fatal. Vaccination is highly effective for prevention.', 'High', 'Caused by Clostridium chauvoei bacteria.'),
    ('High fever, depression, ropey saliva, nasal discharge, sudden death possible', 'Haemorrhagic Septicaemia (HS)', 'Consult Vet Immediately. Specific antibiotics (e.g., Oxytetracycline, Sulphonamides). Vaccination is crucial in endemic areas.', 'High', 'Caused by Pasteurella multocida. Common in monsoon.'),
    ('Blisters/vesicles on tongue, gums, feet (causing lameness), drooling, fever, drop in milk yield', 'Foot-and-Mouth Disease (FMD)', 'Consult Vet & Report. Highly contagious. Supportive care (soft food, antiseptic mouth/foot wash). Strict biosecurity. Vaccination provides protection.', 'High (due to economic impact)', 'Viral disease. Reportable.'),
    ('Swollen, hard, hot, painful udder quarter(s), abnormal milk (clots, watery, bloody), reduced yield, fever possible', 'Mastitis', 'Consult Vet. Intramammary antibiotics based on culture/sensitivity. Frequent milking out. Anti-inflammatories. Prevention via hygiene, proper milking.', 'Medium to High', 'Bacterial infection is common cause.'),
    ('Distended left abdomen (bloat), discomfort, difficulty breathing, kicking at belly, sudden death possible', 'Bloat (Ruminal Tympany)', 'Consult Vet. Emergency. Stomach tube to release gas. Anti-bloat medication (oils, poloxalene). For frothy bloat, trocarization may be needed. Prevent via gradual feed changes.', 'High', 'Frothy (legumes) or free gas bloat.'),
    ('Gradual weight loss despite good appetite, chronic diarrhea, reduced milk yield, bottle jaw (late stage)', 'Johne\'s Disease (Paratuberculosis)', 'Consult Vet. No cure. Test and cull positive animals to control spread. Highly infectious via manure. Long incubation period.', 'Medium (chronic, herd impact)', 'Caused by Mycobacterium avium subspecies paratuberculosis.'),
    ('Fever, anemia (pale gums), jaundice (yellowing), red/dark urine, weakness, tick infestation often present', 'Babesiosis / Theileriosis (Tick Fever)', 'Consult Vet. Specific anti-parasitic drugs (e.g., Diminazene, Buparvaquone). Blood transfusion if severe anemia. Tick control is essential for prevention.', 'Medium to High', 'Protozoan parasites transmitted by ticks.'),
    ('Firm, raised lumps on skin all over body, fever, swollen lymph nodes, nasal/eye discharge, drop in milk yield', 'Lumpy Skin Disease (LSD)', 'Consult Vet. Supportive care (anti-inflammatories, wound care). Antibiotics for secondary bacterial infections. Vector (insect) control helps. Vaccination available.', 'Medium to High', 'Viral disease spread by insects.')
]

inserted_count_disease = 0
for disease_entry in disease_data:
    try:
        cursor.execute('''INSERT INTO disease_diagnosis
                        (symptoms, detected_disease, suggested_treatment, severity, notes)
                        VALUES (?, ?, ?, ?, ?)''', disease_entry)
        inserted_count_disease += 1
    except Exception as e:
        print(f"Error inserting disease entry '{disease_entry[1]}': {e}")

print(f"Disease Diagnosis: Inserted {inserted_count_disease} records.")
connection.commit()

# --- Finalize ---
print("\nClosing database connection.")
connection.close()

print("\nDatabase setup script finished execution.")