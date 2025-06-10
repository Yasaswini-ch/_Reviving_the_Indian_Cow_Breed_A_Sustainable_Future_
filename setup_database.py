import sqlite3
import pandas as pd
import datetime
from datetime import date, timedelta # <--- CORRECTED IMPORT
import random # <--- ADDED IMPORT
import bcrypt # For password hashing
import os # For deleting the old DB if needed

# Database file name (ensure this matches your Streamlit app's DB_FILE)
DB_FILE = 'Cows.db'

def hash_password(password):
    """Hashes a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def create_connection(db_file):
    """Creates a database connection to the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f"SQLite version: {sqlite3.sqlite_version}")
        print(f"Successfully connected to {db_file}")
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
    return conn

def create_table(conn, create_table_sql):
    """Creates a table from the create_table_sql statement."""
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
        # No need to commit here, will commit after all tables or after each group
    except sqlite3.Error as e:
        print(f"Error creating table with SQL:\n{create_table_sql}\nError: {e}")

def main():
    print(f"Database setup script started for: {DB_FILE}")

    # --- DANGER ZONE: Optionally delete the old database file ---
    # --- Uncomment the next 3 lines ONLY if you want a completely fresh database ---
    # --- ALL EXISTING DATA IN Cows.db WILL BE LOST ---
    # if os.path.exists(DB_FILE):
    #     print(f"Deleting existing database file: {DB_FILE} to ensure fresh schema.")
    #     os.remove(DB_FILE)

    conn = create_connection(DB_FILE)

    if conn is not None:
        print("\n--- Creating All Tables (IF NOT EXISTS) ---")

        # --- Core Informational Tables ---

# In setup_cows_db.py
    
        create_table(conn, """
            CREATE TABLE IF NOT EXISTS cattle_breeds (
            breed_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            region TEXT,
            milk_yield INTEGER,
            strength TEXT,
            lifespan INTEGER,
            image_url TEXT,
            symbolism TEXT,    -- ADDED
            scripture TEXT,    -- ADDED
            research TEXT      -- ADDED
        );""")

        create_table(conn, """
        CREATE TABLE IF NOT EXISTS government_schemes (
            scheme_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, details TEXT NOT NULL,
            region TEXT, type TEXT, url TEXT -- Ensured 'type' column
        );""")

        create_table(conn, """
        CREATE TABLE IF NOT EXISTS eco_practices (
            practice_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE, description TEXT,
            category TEXT, suitability TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );""")

        create_table(conn, """
        CREATE TABLE IF NOT EXISTS disease_diagnosis (
            report_id INTEGER PRIMARY KEY AUTOINCREMENT, symptoms TEXT NOT NULL, detected_disease TEXT,
            suggested_treatment TEXT, severity TEXT, notes TEXT, -- Ensured 'severity' and 'notes'
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );""")

        create_table(conn, """
        CREATE TABLE IF NOT EXISTS price_trends (
            trend_id INTEGER PRIMARY KEY AUTOINCREMENT, year INTEGER NOT NULL, month INTEGER NOT NULL,
            breed TEXT, region TEXT, average_price FLOAT, UNIQUE(year, month, breed, region)
        );""")

        # --- User Interaction & AI Tables ---
        create_table(conn, """
        CREATE TABLE IF NOT EXISTS image_analysis (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT, image_path TEXT, uploaded_filename TEXT,
            detected_breed TEXT, confidence_score FLOAT, analysis_backend TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );""")

        create_table(conn, """
        CREATE TABLE IF NOT EXISTS user_queries (
            query_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, user_input TEXT, user_language TEXT,
            translated_input TEXT, bot_response TEXT, response_language TEXT, model_used TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );""")

        # --- User Management & Farm Data Tables ---

        create_table(conn, """
        CREATE TABLE IF NOT EXISTS saved_alerts (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,               -- Buyer's user_id
            alert_name TEXT NOT NULL,               -- User-defined name for the alert
            alert_type TEXT NOT NULL,               -- 'cattle' or 'machinery'
            criteria_json TEXT NOT NULL,            -- JSON string storing search filters
                                                   -- e.g., {"breed": "Gir", "max_price": 50000, "location": "Gujarat"}
                                                   -- e.g., {"machinery_type": "Tractor", "condition": "Used - Good"}
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_checked_at DATETIME DEFAULT NULL,               -- Timestamp of when new results were last checked/shown
            new_matches_count INTEGER DEFAULT 0,    -- Optional: count of new matches since last check
            is_active INTEGER DEFAULT 1,            -- 1 for active, 0 for inactive
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );""")


        create_table(conn, """
        CREATE TABLE IF NOT EXISTS user_cattle (
            cattle_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            tag_id TEXT,
            name TEXT,
            breed TEXT,
            sex TEXT,
            dob DATE,
            purchase_date DATE,
            purchase_price REAL,
            sire_tag_id TEXT,
            dam_tag_id TEXT,
            current_status TEXT DEFAULT 'Active',
            notes TEXT,
            last_calving_date DATE,
            last_heat_observed_date DATE,
            last_insemination_date DATE,
            insemination_sire_tag_id TEXT,
            pregnancy_status TEXT,
            pregnancy_diagnosis_date DATE,
            expected_due_date DATE,
            lactation_number INTEGER DEFAULT 0, -- Ensure this is here
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, tag_id),
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );""")

        
        create_table(conn, """
            CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT, -- Not supplied in INSERT, auto-generated
            username TEXT UNIQUE NOT NULL,            -- 1
            password_hash TEXT NOT NULL,              -- 2
            role TEXT NOT NULL CHECK(role IN ('farmer', 'buyer')), -- 3
            full_name TEXT,                           -- 4
            email TEXT UNIQUE,                        -- 5
            phone_number TEXT,                        -- 6
            address TEXT,                             -- 7
            latitude REAL,                            -- 8
            longitude REAL,                           -- 9
            sells_products INTEGER DEFAULT 0,         -- 10
            product_categories TEXT,                  -- 11
            region TEXT,                              -- 12
            share_contact_info INTEGER DEFAULT 1,     -- 13
            upi_id TEXT,                              -- 14
            registration_date DATETIME DEFAULT CURRENT_TIMESTAMP -- Not supplied, uses DEFAULT
        );""")


        create_table(conn, """
        CREATE TABLE IF NOT EXISTS calf_rearing_log (
            calf_log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            cattle_id INTEGER NOT NULL, -- This is the calf's ID from user_cattle
            log_date DATE NOT NULL,
            colostrum_fed_within_6h BOOLEAN, -- Checkbox: Yes/No
            colostrum_amount_liters REAL,
            milk_replacer_type TEXT,
            milk_replacer_amount_liters REAL,
            starter_feed_intake_grams REAL,
            deworming_done BOOLEAN,
            deworming_product TEXT,
            vaccination_given TEXT, -- e.g., "Initial BQ Vaccine"
            weight_kg REAL,
            health_notes TEXT, -- e.g., "Slight scour, gave electrolytes"
            FOREIGN KEY (cattle_id) REFERENCES user_cattle(cattle_id) ON DELETE CASCADE
        );""")
        
        create_table(conn, """
        CREATE TABLE IF NOT EXISTS vaccination_records (
            vaccination_id INTEGER PRIMARY KEY AUTOINCREMENT,
            cattle_id INTEGER NOT NULL,
            vaccine_name TEXT NOT NULL,
            vaccination_date DATE NOT NULL,
            booster_due_date DATE,
            administered_by TEXT,
            batch_number TEXT,
            notes TEXT,
            FOREIGN KEY (cattle_id) REFERENCES user_cattle(cattle_id) ON DELETE CASCADE
        );""")

        # --- NEW Health & Vaccination Log Tables ---
        create_table(conn, """
        CREATE TABLE IF NOT EXISTS vaccinations_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            cattle_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL, -- To link back to the user easily
            vaccination_name TEXT NOT NULL,
            vaccination_date DATE NOT NULL,
            next_due_date DATE, -- For scheduling boosters/annual
            batch_number TEXT,
            administered_by TEXT, -- Vet name or 'Self'
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (cattle_id) REFERENCES user_cattle(cattle_id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );""")

        create_table(conn, """
        CREATE TABLE IF NOT EXISTS health_records (
            health_record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            cattle_id INTEGER NOT NULL,
            record_date DATE NOT NULL,
            symptoms_observed TEXT,
            diagnosis TEXT,
            treatment_given TEXT,
            vet_name TEXT,
            cost REAL,
            outcome TEXT,
            notes TEXT,
            FOREIGN KEY (cattle_id) REFERENCES user_cattle(cattle_id) ON DELETE CASCADE
        );""")

        create_table(conn, """
        CREATE TABLE IF NOT EXISTS health_events_log (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            cattle_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL, -- To link back to the user easily
            event_type TEXT NOT NULL, -- e.g., 'Illness', 'Treatment', 'Routine Checkup', 'Deworming'
            event_date DATE NOT NULL,
            symptoms_observed TEXT,
            diagnosis TEXT,
            treatment_administered TEXT,
            veterinarian_involved TEXT,
            next_checkup_date DATE, -- For scheduling follow-ups
            outcome TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (cattle_id) REFERENCES user_cattle(cattle_id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );""")

        # --- NEW Health Reminders Status Table ---
        create_table(conn, """
        CREATE TABLE IF NOT EXISTS health_reminders_status (
            reminder_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            cattle_id INTEGER NOT NULL,
            original_log_id INTEGER NOT NULL, -- ID from vaccinations_log or health_events_log
            reminder_type TEXT NOT NULL,      -- 'vaccination_due', 'health_checkup_due'
            reminder_description TEXT NOT NULL,
            due_date DATE NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', -- 'pending', 'completed', 'dismissed'
            action_taken_on TIMESTAMP,
            action_notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
            FOREIGN KEY (cattle_id) REFERENCES user_cattle(cattle_id) ON DELETE CASCADE
        );""")
        # Indexes for health_reminders_status
        create_table(conn, "CREATE INDEX IF NOT EXISTS idx_hrs_user_status_due ON health_reminders_status (user_id, status, due_date);")
        create_table(conn, "CREATE INDEX IF NOT EXISTS idx_hrs_cattle_status_due ON health_reminders_status (cattle_id, status, due_date);")

                # --- Marketplace Tables ---
        create_table(conn, """
        CREATE TABLE IF NOT EXISTS cattle_for_sale (
            listing_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,         -- Seller's user_id
            cattle_id INTEGER UNIQUE NOT NULL, -- The cattle_id from user_cattle table
            asking_price REAL NOT NULL,
            listing_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            description TEXT,
            location TEXT,                    -- Listing specific location
            status TEXT DEFAULT 'Available',  -- Available, Sold, Withdrawn
            views_count INTEGER DEFAULT 0,
            sold_date DATETIME,
            buyer_id INTEGER,
            image_url_1 TEXT, -- <<<< Was added here
            image_url_2 TEXT, -- <<<< Was added here
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
            FOREIGN KEY (cattle_id) REFERENCES user_cattle(cattle_id) ON DELETE CASCADE,
            FOREIGN KEY (buyer_id) REFERENCES users(user_id) ON DELETE SET NULL
        );""")


        create_table(conn, """
        CREATE TABLE IF NOT EXISTS machinery_listings (
            machinery_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,      -- Seller's user_id
            name TEXT NOT NULL,
            type TEXT,
            brand TEXT,
            model TEXT,
            year_of_manufacture INTEGER,
            condition TEXT,
            asking_price REAL,
            description TEXT,
            location TEXT,
            listing_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'Available', -- Available, Sold, Withdrawn
            image_url_1 TEXT,
            image_url_2 TEXT,
            for_rent INTEGER DEFAULT 0,    -- 0 for No, 1 for Yes
            rental_price_day REAL,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );""")

        # --- Breeding & Nutrition Tables ---
        create_table(conn, """
        CREATE TABLE IF NOT EXISTS breeding_pairs (
            pair_id INTEGER PRIMARY KEY AUTOINCREMENT, cattle_1 TEXT, cattle_2 TEXT, goal TEXT,
            genetic_score INTEGER, status TEXT, notes TEXT, -- Ensured 'notes', 'genetic_score', 'status'
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );""")

        create_table(conn, """CREATE TABLE IF NOT EXISTS veterinarians ( vet_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, clinic_name TEXT, specialization TEXT, address TEXT NOT NULL, city TEXT, state TEXT, pincode TEXT, phone_number TEXT, email TEXT, latitude REAL NOT NULL, longitude REAL NOT NULL, services_offered TEXT, operating_hours TEXT, is_verified INTEGER DEFAULT 0);""")

        create_table(conn, """
        CREATE TABLE IF NOT EXISTS offspring_data (
            offspring_data_id INTEGER PRIMARY KEY AUTOINCREMENT, parent_1 TEXT, parent_2 TEXT,
            offspring_id TEXT UNIQUE, breed TEXT, sex TEXT, dob DATE, predicted_traits TEXT,
            actual_traits TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );""")

        create_table(conn, """
        CREATE TABLE IF NOT EXISTS user_saved_listings (
            saved_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,       -- User who saved it
            listing_type TEXT NOT NULL,     -- 'cattle' or 'machinery'
            original_listing_id INTEGER NOT NULL, -- The listing_id from cattle_for_sale or machinery_id from machinery_listings
            saved_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            notes TEXT,                     -- Optional user notes for this saved item
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
            UNIQUE (user_id, listing_type, original_listing_id) -- User can save a specific listing only once
        );""")

        create_table(conn, """
        CREATE TABLE IF NOT EXISTS feedstuffs (
            feedstuff_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            category TEXT,
            dm_percent REAL,
            cp_percent REAL,
            tdn_percent REAL, -- Added TDN as it's a common measure
            me_mj_kg_dm REAL,
            ca_percent REAL,
            p_percent REAL,
            price_per_kg_estimate REAL, -- Added for economic evaluation
            notes TEXT,
            region_suitability TEXT -- Added for location-based suggestions
        );""") # Added new columns to feedstuffs
        
        
# --- INSIDE main() in setup_cows_db.py, with other CREATE TABLE statements ---

        create_table(conn, """
        CREATE TABLE IF NOT EXISTS impact_campaigns (
            campaign_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            goal_amount REAL, -- Optional: target amount
            current_amount REAL DEFAULT 0,
            start_date DATE,
            end_date DATE,
            status TEXT DEFAULT 'Active', -- Active, Completed, Archived
            image_url TEXT, -- An image for the campaign
            category TEXT -- e.g., Fodder, Vet Care, Farmer Training, Gaushala Support
        );""")

        create_table(conn, """
    CREATE TABLE IF NOT EXISTS donations_log (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        campaign_id INTEGER,
        adoptable_animal_id INTEGER,
        amount REAL NOT NULL,
        donor_name TEXT,
        payment_method TEXT,
        payment_status TEXT NOT NULL, -- e.g., 'Pledged', 'Completed', 'Failed'
        transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_recurring BOOLEAN DEFAULT 0, -- 1 for recurring sponsorships (like monthly adoption), 0 for one-time
        FOREIGN KEY (user_id) REFERENCES users(user_id),
        FOREIGN KEY (campaign_id) REFERENCES impact_campaigns(campaign_id),
        FOREIGN KEY (adoptable_animal_id) REFERENCES adoptable_animals(adoptable_animal_id)
    );""")

        create_table(conn, """
        CREATE TABLE IF NOT EXISTS adoptable_animals (
            adoptable_animal_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            species TEXT DEFAULT 'Cow', -- Cow, Calf, Bull
            breed TEXT,
            sex TEXT,
            approx_age_years REAL, -- Can be in years or months
            story TEXT, -- Their background story
            health_status TEXT,
            image_url_1 TEXT,
            image_url_2 TEXT,
            location_info TEXT, -- e.g., "Sri Krishna Gaushala, Nagpur"
            monthly_sponsorship_cost REAL, -- Estimated cost to care for them per month
            is_adopted INTEGER DEFAULT 0, -- 0 for No, 1 for Yes
            gaushala_or_org_name TEXT, -- Name of the Gaushala or organization managing the adoption
            contact_for_adoption TEXT -- Email or phone for adoption inquiries
        );""")

        create_table(conn, """
CREATE TABLE IF NOT EXISTS adoptions_log (
    adoption_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL, -- Adopter's user_id
    adoptable_animal_id INTEGER NOT NULL,
    adoption_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    adoption_type TEXT, -- e.g., 'Full Sponsorship', 'Partial Monthly'
    duration_months INTEGER, -- If applicable
    notes TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (adoptable_animal_id) REFERENCES adoptable_animals(adoptable_animal_id) ON DELETE RESTRICT
);""")
        
        create_table(conn, """
    CREATE TABLE IF NOT EXISTS offspring_data (
        offspring_data_id INTEGER PRIMARY KEY AUTOINCREMENT,
        parent_1 TEXT,
        parent_2 TEXT,
        offspring_id TEXT UNIQUE,
        breed TEXT,
        sex TEXT,
        dob DATE,
        predicted_traits TEXT,
        actual_traits TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );""")
        

        

        create_table(conn, """
        CREATE TABLE IF NOT EXISTS milk_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            cattle_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            log_date TEXT NOT NULL,
            milking_session TEXT CHECK(milking_session IN ('Morning', 'Evening', 'Afternoon', 'Full Day', 'Other')),
            milk_yield_liters REAL NOT NULL,
            fat_percentage REAL,
            snf_percentage REAL,
            notes TEXT,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (cattle_id) REFERENCES user_cattle(cattle_id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );""")
        create_table(conn, "CREATE INDEX IF NOT EXISTS idx_milk_log_cattle_date ON milk_log (cattle_id, log_date);")
        
        # --- INSIDE main() in setup_cows_db.py, with other CREATE TABLE statements ---
        create_table(conn, """
            CREATE TABLE IF NOT EXISTS indicative_prices (
            price_guide_id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_type TEXT NOT NULL,          -- 'Cattle' or 'Machinery'
            breed_or_machinery_type TEXT,     -- e.g., 'Gir', 'Sahiwal', 'Tractor', 'Thresher'
            category_subtype TEXT,            -- e.g., 'Milking Cow (2nd Lactation)', 'Heifer (Breeding Age)', '35 HP Used'
            region TEXT,                      -- e.g., 'Gujarat', 'Punjab', 'North India', 'All India'
            price_range_low REAL,
            price_range_high REAL,
            currency TEXT DEFAULT 'INR',
            data_source TEXT,                 -- e.g., 'Expert Estimate 2024', 'Local Market Survey Q1 2024'
            notes TEXT,                       -- Any additional notes about this price range
            last_updated DATE
        );""")

        
    

        conn.commit() # Commit all table creations
        print("All tables checked/created successfully.")

        # --- Insert Sample Data ---
        print("\n--- Inserting Sample Data (IF NOT EXISTS) ---")
        cursor = conn.cursor()
    
        # 1. Users
        users_data = [
    # username, password, role, full_name, email, phone, address, region, lat, long, sells_prod, prod_cat, share_contact
        ('farmer_john', hash_password('pass123'), 'farmer', 'John Doe', 'john@farm.com', '+1234567890', '123 Farm Road, Villagetown', 'Maharashtra', 18.5204, 73.8567, 1, 'Milk,Ghee', 1, 'johns_upi@oksbi'),
        ('buyer_jane', hash_password('pass123'), 'buyer', 'Jane Smith', 'jane@buy.com', '+0987654321', '456 Market St, Cityville', 'Gujarat', None, None, 0, None, 0, None),
        ('farmer_kumar', hash_password('kumar123'), 'farmer', 'Kumar Patel', 'kumar@village.net', None, 'Plot 5, Rural Area', 'Punjab', 30.7333, 76.7794, 1, 'Dung Cakes,Urine', 0, None), # Chandigarh coords
        #('buyer_jane', hash_password('pass123'), 'buyer', 'Jane Smith', 'jane@buy.com', '+0987654321', '456 Market St, Cityville', 'Gujarat', None, None, 0, None, 1, None),
        ('agri_services', hash_password('agri123'), 'buyer', 'Agri Services Ltd.', 'contact@agriserv.com', None, '789 Industrial Area', 'Karnataka', None, None, 0, None, 1,"agri_upi@ybl")
        ]
        inserted_count = 0
        for u_data in users_data:
            try:
                cursor.execute("""
                    INSERT INTO users (
                        username, password_hash, role, full_name, email,
                        phone_number, address, region, latitude, longitude,
                        sells_products, product_categories, share_contact_info, upi_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, u_data) # u_data tuple must have 14 items
                # --- END OF SQL STATEMENT TO CHECK ---
                if cursor.rowcount > 0: inserted_count += 1
            except sqlite3.IntegrityError: pass
            except Exception as e: print(f"Error inserting user {u_data[0]}: {e}")
        print(f"Users: Inserted {inserted_count} new records.")
        conn.commit()

        user_ids = {}
        try:
            cursor.execute("SELECT username, user_id FROM users")
            for row in cursor.fetchall():
                user_ids[row[0]] = row[1]
        except Exception as e:
            print(f"Could not fetch user_ids for sample data insertion: {e}")

        # 2. User Cattle (My Herd)
        if 'farmer_john' in user_ids and 'farmer_kumar' in user_ids:
            user_cattle_data = [
                (user_ids['farmer_john'], 'MH01-001', 'Laxmi', 'Gir', 'Cow', '2019-04-10', '2021-01-01', 55000.0, 'GIR-S1', 'GIR-D1', 'Active', 'Good milker.'),
                (user_ids['farmer_john'], 'MH01-002', 'Gauri', 'Sahiwal', 'Heifer', '2022-08-15', '2022-08-15', 30000.0, None, 'SAHI-D2', 'Active', 'First calf expected next year.'),
                (user_ids['farmer_kumar'], 'PB01-101', 'Raja', 'Murrah', 'Bull', '2020-01-20', '2020-06-01', 80000.0, None, None, 'Active', 'Proven breeding bull.')
            ]
            inserted_count = 0
            for uc_data in user_cattle_data:
                try:
                    cursor.execute("INSERT INTO user_cattle (user_id, tag_id, name, breed, sex, dob, purchase_date, purchase_price, sire_tag_id, dam_tag_id, current_status, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", uc_data)
                    if cursor.rowcount > 0: inserted_count += 1
                except sqlite3.IntegrityError: pass
                except Exception as e: print(f"Error inserting user_cattle: {e}")
            print(f"User Cattle: Inserted {inserted_count} new records.")
            conn.commit()

        # 3. Cattle For Sale (Marketplace)
        if 'farmer_john' in user_ids:
            try:
                cursor.execute("SELECT cattle_id FROM user_cattle WHERE user_id = ? AND tag_id = 'MH01-001'", (user_ids['farmer_john'],))
                laxmi_cattle_id_row = cursor.fetchone()
                if laxmi_cattle_id_row:
                    laxmi_cattle_id = laxmi_cattle_id_row[0]
                    cattle_for_sale_data = [
                        (user_ids['farmer_john'], laxmi_cattle_id, 65000.0, 'Healthy Gir cow, 2nd lactation, yields 10L/day.', 'Villagetown, Maharashtra', 'Available')
                    ]
                    inserted_count = 0
                    for cfs_data in cattle_for_sale_data:
                        try:
                            cursor.execute("INSERT INTO cattle_for_sale (user_id, cattle_id, asking_price, description, location, status) VALUES (?, ?, ?, ?, ?, ?)", cfs_data)
                            cursor.execute("UPDATE user_cattle SET current_status = 'For Sale' WHERE cattle_id = ?", (cfs_data[1],))
                            if cursor.rowcount > 0: inserted_count += 1
                        except sqlite3.IntegrityError: pass
                        except Exception as e: print(f"Error inserting cattle_for_sale: {e}")
                    print(f"Cattle For Sale: Inserted {inserted_count} new records.")
                    conn.commit()
            except Exception as e:
                print(f"Error preparing cattle_for_sale sample data: {e}")

        # 4. Machinery Listings
        if 'farmer_john' in user_ids and 'agri_services' in user_ids and 'farmer_kumar' in user_ids:
            machinery_data = [
                (user_ids['farmer_john'], 'Used Tractor', 'Tractor', 'Mahindra', '265 DI', 2015, 'Used - Good', 250000.0, 'Well-maintained, 35 HP, ready for work.', 'Villagetown, Maharashtra', None, None, 0, None),
                (user_ids['agri_services'], 'Power Tiller (New)', 'Tillage', 'VST Shakti', '130 DI', datetime.date.today().year, 'New', 85000.0, 'Brand new power tiller, 13HP, with warranty.', 'Cityville, Karnataka', None, None, 1, 1200.0),
                (user_ids['farmer_kumar'], 'Old Plough', 'Tillage', 'Local Make', None, 2005, 'Used - Fair', 5000.0, 'Basic plough, needs some repair but functional.', 'Rural Area, Punjab', None, None, 0, None)
            ]
            inserted_count = 0
            for m_data in machinery_data:
                try:
                    cursor.execute("INSERT INTO machinery_listings (user_id, name, type, brand, model, year_of_manufacture, condition, asking_price, description, location, image_url_1, image_url_2, for_rent, rental_price_day) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", m_data)
                    if cursor.rowcount > 0: inserted_count += 1
                except Exception as e: print(f"Error inserting machinery: {e}")
            print(f"Machinery Listings: Inserted {inserted_count} new records.")
            conn.commit()

        # 5. Feedstuffs
        print("\n--- Processing Feedstuffs ---")
        feedstuffs_sample_data = [
            # Name, Category, DM_%, CP_%, TDN_%, ME_MJ_kg_DM, Ca_%, P_%, Price_Estimate, Notes, Region
            ('Maize Grain (Corn)', 'Concentrate-Energy', 88.0, 9.0, 85.0, 12.5, 0.02, 0.28, 25.0, 'Good energy source', 'All India'),
            ('Wheat Bran', 'Concentrate-Energy', 89.0, 15.0, 67.0, 10.0, 0.12, 1.18, 20.0, 'Palatable, good for fiber', 'All India'),
            ('Soybean Meal', 'Concentrate-Protein', 90.0, 44.0, 78.0, 12.0, 0.30, 0.65, 45.0, 'Excellent protein source', 'All India'),
            ('Groundnut Cake (GNC)', 'Concentrate-Protein', 90.0, 45.0, 75.0, 11.5, 0.20, 0.60, 40.0, 'Common protein supplement', 'All India'),
            ('Mustard Cake', 'Concentrate-Protein', 90.0, 35.0, 70.0, 10.5, 0.60, 1.00, 30.0, 'Use with caution, can be bitter', 'North India'),
            ('Cotton Seed Cake (Undec.)', 'Concentrate-Protein', 92.0, 22.0, 65.0, 9.8, 0.16, 0.90, 32.0, 'Contains gossypol, limit use for young/breeding stock', 'All India'),
            ('Berseem (Green Fodder)', 'Green Fodder', 20.0, 18.0, 60.0, 9.0, 1.50, 0.30, 3.0, 'Excellent leguminous green fodder', 'North India (Rabi)'),
            ('Lucerne (Alfalfa - Green)', 'Green Fodder', 22.0, 20.0, 62.0, 9.3, 1.40, 0.25, 4.0, 'High protein green fodder', 'All India (where adapted)'),
            ('Napier Bajra Hybrid (Green)', 'Green Fodder', 18.0, 10.0, 55.0, 8.0, 0.50, 0.30, 2.0, 'High yielding grass', 'All India (Tropical/Subtropical)'),
            ('Sorghum Fodder (Jowar - Green)', 'Green Fodder', 25.0, 8.0, 58.0, 8.5, 0.40, 0.25, 2.5, 'Good summer fodder, watch for HCN in early stages', 'All India'),
            ('Maize Fodder (Green)', 'Green Fodder', 22.0, 9.0, 60.0, 9.0, 0.35, 0.22, 2.8, 'Palatable green fodder', 'All India'),
            ('Wheat Straw (Bhusa)', 'Dry Fodder', 90.0, 3.5, 40.0, 5.5, 0.22, 0.10, 8.0, 'Low quality roughage, needs supplementation', 'All India'),
            ('Paddy Straw', 'Dry Fodder', 90.0, 3.0, 38.0, 5.0, 0.20, 0.08, 6.0, 'Very low quality, needs urea treatment/supplementation', 'All India (Rice areas)'),
            ('Groundnut Haulms (Dry)', 'Dry Fodder', 88.0, 10.0, 50.0, 7.5, 1.00, 0.18, 10.0, 'Good quality leguminous dry fodder', 'Groundnut growing areas'),
            ('Mineral Mixture (Generic Dairy)', 'Mineral Mix', 98.0, 0.0, 0.0, 0.0, 20.0, 10.0, 60.0, 'Essential for balancing minerals', 'All India'),
            ('Common Salt', 'Feed Additive', 99.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 'Sodium and Chloride source', 'All India')
        ]
        inserted_count_feed = 0
        skipped_count_feed = 0
        for feed in feedstuffs_sample_data:
            try:
                cursor.execute('''
                    INSERT INTO feedstuffs (name, category, dm_percent, cp_percent, tdn_percent, me_mj_kg_dm, ca_percent, p_percent, price_per_kg_estimate, notes, region_suitability)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', feed)
                if cursor.rowcount > 0: inserted_count_feed += 1
            except sqlite3.IntegrityError: skipped_count_feed += 1
            except Exception as e:
                print(f"Error inserting feedstuff {feed[0]}: {e}")
                skipped_count_feed += 1
        print(f"Feedstuffs: Inserted {inserted_count_feed}, Skipped {skipped_count_feed} duplicates.")
        conn.commit()

        # --- Populate Original Tables (ensure data matches expected structure) ---
        print("\n--- Processing Government Schemes (Sample) ---")
        schemes_sample = [ # This is your full list of schemes
            ('Rashtriya Gokul Mission', 'Promotes indigenous cattle breeding and genetic improvement.', 'All India / Central', 'Animal Husbandry', 'https://dahd.nic.in/schemes/programmes/rashtriya-gokul-mission'),
            ('National Livestock Mission (NLM)', 'Sustainable development of livestock sector, covering feed/fodder, breed improvement, entrepreneurship.', 'All India / Central', 'Animal Husbandry', 'https://dahd.nic.in/nlm'),
            ('Dairy Entrepreneurship Development Scheme (DEDS - aspects now in DIDF)', 'Financial support for setting up small dairy farms & units (Check NABARD/NDDB for current alternatives like DIDF).', 'All India / Central', 'Dairy Development', 'https://www.nabard.org/content1.aspx?id=591'),
            ('Kisan Credit Card (KCC) Scheme', 'Provides short-term credit to farmers for agriculture and allied activities including animal husbandry.', 'All India / Central', 'Finance/Credit', 'https://pmkisan.gov.in/kcc/'),
            ('PM-KUSUM', 'Promotes solar energy use in agriculture, including solar pumps and potentially solar power for dairy farms/biogas plants.', 'All India / Central', 'Energy/Agriculture', 'https://pmkusum.mnre.gov.in/'),
            ('National Programme for Dairy Development (NPDD)', 'Aims to strengthen dairy cooperatives and increase milk production.', 'All India / Central', 'Dairy Development', 'https://dahd.nic.in/npdd'),
            ('Livestock Health & Disease Control (LH&DC) Programme', 'Focuses on prevention, control and eradication of animal diseases, including FMD, Brucellosis.', 'All India / Central', 'Animal Health', 'https://dahd.nic.in/lh-dc'),
            ('Animal Husbandry Infrastructure Development Fund (AHIDF)', 'Incentivizes investments in dairy processing, value addition infrastructure, and animal feed plants.', 'All India / Central', 'Infrastructure', 'https://ahidf.udyamimitra.in/'),
            ('Gobar Dhan Scheme', 'Promotes converting cattle dung and solid waste into compost, biogas, and biofuel.', 'All India / Central', 'Waste Management/Energy', 'https://sbm.gov.in/Gobardhan/'),
            ('Mukhyamantri Dugdh Utpadak Sambal Yojana (Rajasthan)', 'Provides bonus per litre of milk to cooperative dairy farmers.', 'Rajasthan', 'Subsidy/Incentive', 'https://animalhusbandry.rajasthan.gov.in/'),
            ('Mukhyamantri Gau Mata Poshan Yojana (Gujarat)', 'Assistance for maintenance of unproductive/old cattle in Gaushalas/Panjrapoles.', 'Gujarat', 'Animal Welfare', 'https://cmogujarat.gov.in/en/latest-news/greeting-ceremony-cm-announcing-mukhyamantri-gaumata-poshan-yojana'),
            ('Ksheera Santhwanam (Kerala)', 'Insurance scheme for dairy farmers covering cattle death.', 'Kerala', 'Insurance/Welfare', 'https://ksheerasree.kerala.gov.in/'),
            ('Nand Baba Milk Mission (Uttar Pradesh)', 'Aims to enhance dairy infrastructure and market access for milk producers.', 'Uttar Pradesh', 'Dairy Development', 'https://updairydevelopment.gov.in/'),
        ]
        inserted_count_gs = 0
        skipped_count_gs = 0
        for scheme_info in schemes_sample:
            try:
                cursor.execute("INSERT INTO government_schemes (name, details, region, type, url) VALUES (?, ?, ?, ?, ?)", scheme_info)
                if cursor.rowcount > 0: inserted_count_gs += 1
            except sqlite3.IntegrityError: skipped_count_gs += 1
        print(f"Government Schemes: Inserted {inserted_count_gs}, Skipped {skipped_count_gs} duplicates.")
        conn.commit()

        print("\n--- Processing Cattle Breeds (Sample) ---")
        CATTLE_BREEDS_SAMPLE_DATA = [
            ("Gir", "Gujarat", 12, "High", 18, "images/gir.jpg",
            "🙏 Symbol of prosperity and abundance in Gujarat.",
            "🕉️ Mentioned in Rigveda as divine nurturer.",
            "🧪 NDRI studies show high heat resistance and good milk yield genetics."),

            ("Sahiwal", "Punjab", 14, "Medium", 20, "images/sahiwal.jpg",
            "🍃 Represents nourishment and resilience in north-western India.",
            "📜 Known in folklore as the cow of fertility and productivity.",
            "🧬 IARI research explores high lactation and disease resistance."),

            ("Ongole", "Andhra Pradesh", 10, "Very High", 22, "images/ongole.jpg",
            "💪 Known as 'Nelore' abroad, symbolizes strength and endurance.",
            "🕉️ Revered in Puranas for warrior-like might in battlefields.",
            "🔬 Global research on its genetics for meat and bullock strength."),

            ("Punganur", "Andhra Pradesh", 6, "Low", 15, "images/punganur.jpg",
            "🍼 Sacred dwarf breed, cherished in temple rituals.",
            "📚 Referenced in South Indian temple inscriptions.",
            "🔎 Studies show efficient metabolism and low feed requirements."),

            ("Amrit Mahal", "Karnataka", 7, "High", 18, "images/amritmahal.jpg",
            "🏹 Historical war cattle of Mysore kings, symbol of valor.",
            "📖 Linked to royal herds and ancient cavalry texts.",
            "🧪 Evaluated for stamina and performance in hilly terrains."),

            ("Deoni", "Maharashtra", 9, "Medium", 19, "images/deoni.jpeg",
            "🌾 Represents dual-purpose utility: milk and plough.",
            "📜 Mentioned in Maratha farming chronicles.",
            "🔬 Studied for balanced lactation and drought resistance."),

            ("Hallikar", "Karnataka", 8, "Very High", 20, "images/hallikar.jpg",
            "🛡️ Signifies warrior traits in southern Indian culture.",
            "🕉️ Cited in ancient Kannada scriptures as sacred draft animal.",
            "🔍 Research highlights speed and load capacity."),

            ("Kankrej", "Gujarat", 11, "High", 21, "images/kankrej.jpg",
            "🕌 Blend of strength and beauty, honored in Gujarat folklore.",
            "📚 Part of Harappan civilization relics as bull sculptures.",
            "🧬 Genetic studies show adaptability to harsh climates."),

            ("Krishna Valley", "Karnataka", 7, "Very High", 19, "images/krishna_valley.jpg",
            "🌊 Represents abundance of Krishna River plains.",
            "📖 Local myth links it to Krishna's cows.",
            "🧪 Trials for high fodder conversion efficiency."),

            ("Malnad Gidda", "Karnataka", 5, "Medium", 16, "images/malnad_gidda.jpeg",
             "🍀 Sacred free-range hill cattle, linked to forest deities.",
             "📜 Cited in folk songs and tribal traditions.",
             "🧫 Known for medicinal value in milk and urine."),

            ("Rathi", "Rajasthan", 10, "Medium", 20, "images/rathi.jpg",
             "🌵 Known as desert dairy queen, symbol of sustenance.",
             "📜 Mentioned in Rajasthani ballads and oral traditions.",
             "🧬 NDRI breeding programs show high yield potential."),

            ("Red Sindhi", "Sindh (Origin)", 11, "High", 22, "images/red_sindhi.jpg",
             "🔥 Represents fertility and robust adaptability.",
             "📖 Revered in ancient Sindhi rituals.",
             "🔬 Popular in international dairy research for cross-breeding."),

            ("Tharparkar", "Rajasthan", 9, "Medium", 21, "images/tharparkar.jpg",
             "🌞 Survives in extreme heat, revered for endurance.",
             "📜 Known in Vedic contexts as protector of the herds.",
             "🧪 Heat tolerance studies by Indian Vet Research Institute.")
        ]

        inserted_count_cb = 0
        skipped_count_cb = 0

        # Each breed_item_tuple now has 9 elements
        for breed_item_tuple in CATTLE_BREEDS_SAMPLE_DATA:
            try:
                cursor.execute("""
                               INSERT INTO cattle_breeds (name, region, milk_yield, strength, lifespan, image_url, symbolism, scripture, research)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", breed_item_tuple)
                if cursor.rowcount > 0:
                    inserted_count_cb += 1
            except sqlite3.IntegrityError:
                skipped_count_cb += 1
            except Exception as e:
                print(f"Error inserting cattle breed {breed_item_tuple[0] if breed_item_tuple else 'Unknown'}: {e}")
                skipped_count_cb += 1
        print(f"Cattle Breeds: Inserted {inserted_count_cb}, Skipped {skipped_count_cb} duplicates.")
        conn.commit()           


        print("\n--- Processing Breeding Pairs (Sample) ---")
        breeding_pairs_sample_data = [
            ('GIR-BULL-01', 'GIR-COW-A5', 'High Milk Yield', 85, 'Suggested', 'Good match for milk traits.', (datetime.datetime.now() - datetime.timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S")),
            ('SAH-BULL-03', 'SAH-COW-B2', 'Breed Purity', 92, 'Suggested', 'Excellent lineage match.', (datetime.datetime.now() - datetime.timedelta(days=8)).strftime("%Y-%m-%d %H:%M:%S")),
        ]
        inserted_count_bp = 0
        for pair_info in breeding_pairs_sample_data:
            try:
                cursor.execute('''INSERT INTO breeding_pairs (cattle_1, cattle_2, goal, genetic_score, status, notes, timestamp)
                                  VALUES (?, ?, ?, ?, ?, ?, ?)''', pair_info)
                if cursor.rowcount > 0: inserted_count_bp += 1
            except Exception as e: print(f"Error inserting breeding pair: {e}")
        print(f"Breeding Pairs: Inserted {inserted_count_bp} new records.")
        conn.commit()

        print("\n--- Processing Offspring Data (Sample) ---")
        offspring_sample_data = [
    ('RATHI-BULL-R2', 'RATHI-COW-D1', 'RATHI-CALF-001', 'Rathi', 'Female', '2023-11-15', 'Good confirmation', 'Developing well', (datetime.datetime.now() - datetime.timedelta(days=25)).strftime("%Y-%m-%d %H:%M:%S")),
    ('GIR-BULL-G4', 'GIR-COW-G2', 'GIR-CALF-002', 'Gir', 'Male', '2023-10-20', 'Excellent traits', 'Growing steadily', (datetime.datetime.now() - datetime.timedelta(days=60)).strftime("%Y-%m-%d %H:%M:%S")),
    ('SAHIWAL-BULL-S1', 'SAHIWAL-COW-S3', 'SAHIWAL-CALF-003', 'Sahiwal', 'Female', '2023-09-15', 'Robust physique', 'Energetic', (datetime.datetime.now() - datetime.timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S")),
    ('THARPARKAR-BULL-T2', 'THARPARKAR-COW-T4', 'THARPARKAR-CALF-004', 'Tharparkar', 'Male', '2023-08-10', 'Good dairy potential', 'Healthy growth', (datetime.datetime.now() - datetime.timedelta(days=120)).strftime("%Y-%m-%d %H:%M:%S")),
    ('ONGOLE-BULL-O5', 'ONGOLE-COW-O7', 'ONGOLE-CALF-005', 'Ongole', 'Female', '2023-07-25', 'Hardy and strong', 'Well-developed', (datetime.datetime.now() - datetime.timedelta(days=150)).strftime("%Y-%m-%d %H:%M:%S")),
    ('KANKREJ-BULL-K3', 'KANKREJ-COW-K8', 'KANKREJ-CALF-006', 'Kankrej', 'Male', '2023-06-30', 'Fast growth rate', 'Active and sturdy', (datetime.datetime.now() - datetime.timedelta(days=180)).strftime("%Y-%m-%d %H:%M:%S")),
]

        inserted_count_od = 0
        skipped_count_od = 0
        for off_data in offspring_sample_data:
            try:
                cursor.execute('''INSERT INTO offspring_data (parent_1, parent_2, offspring_id, breed, sex, dob, predicted_traits, actual_traits, timestamp)
                                  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', off_data)
                if cursor.rowcount > 0: inserted_count_od += 1
            except sqlite3.IntegrityError: skipped_count_od += 1
        print(f"Offspring Data: Inserted {inserted_count_od}, Skipped {skipped_count_od} duplicates.")
        conn.commit()

        print("\n--- Processing Eco Practices (Sample) ---")
        eco_practices_sample_data = [
            ('Manure Composting', 'Decomposing manure with crop residues to create rich organic fertilizer.', 'Manure Management', 'Cattle Farms'),
            ('Biogas Production', 'Anaerobic digestion of dung to produce cooking gas and slurry.', 'Manure Management/Energy', 'Cattle Farms'),
        ]
        inserted_count_ep = 0
        skipped_count_ep = 0
        for eco_prac in eco_practices_sample_data:
            try:
                cursor.execute("INSERT INTO eco_practices (name, description, category, suitability) VALUES (?, ?, ?, ?)", eco_prac)
                if cursor.rowcount > 0: inserted_count_ep += 1
            except sqlite3.IntegrityError: skipped_count_ep += 1
        print(f"Eco Practices: Inserted {inserted_count_ep}, Skipped {skipped_count_ep} duplicates.")
        conn.commit()


        print("\n--- Processing Price Trends (Sample) ---")
        price_trends_sample_data = [
            (2023, 10, 'Gir', 'Gujarat', 65000), (2023, 10, 'Sahiwal', 'Punjab', 68000),
            (2024, 1, 'Crossbred', 'Maharashtra', 56000),
        ]
        inserted_count_pt = 0
        skipped_count_pt = 0
        for trend_data in price_trends_sample_data:
            try:
                cursor.execute("INSERT INTO price_trends (year, month, breed, region, average_price) VALUES (?, ?, ?, ?, ?)", trend_data)
                if cursor.rowcount > 0: inserted_count_pt += 1
            except sqlite3.IntegrityError: skipped_count_pt +=1
        print(f"Price Trends: Inserted {inserted_count_pt}, Skipped {skipped_count_pt} duplicates.")
        conn.commit()

        print("\n--- Processing Veterinarians (Sample) ---")
        veterinarians_data = [
            ('Dr. Anita Sharma', 'Pashu Seva Clinic', 'Large Animal', '10 Main Street, Ruralville', 'Ruralville', 'Maharashtra', '411001', '9876500001', 'anita.sharma@vet.com', 18.5210, 73.8570, 'Consultation,Vaccination,AI', 'Mon-Sat: 10AM-5PM', 1),
            ('Dr. Rajesh Singh', 'Cattle Care Center', 'Bovine Specialist', 'Highway Road, Farmtown', 'Farmtown', 'Punjab', '141001', '9876500002', 'rajesh.singh@vetclinic.com', 30.7340, 76.7800, 'Consultation,Surgery,Emergency', '24/7 Emergency, Mon-Fri: 9AM-7PM', 1),
        ]
        inserted_vets = 0
        for vet_data in veterinarians_data:
            try:
                cursor.execute("INSERT INTO veterinarians (name, clinic_name, specialization, address, city, state, pincode, phone_number, email, latitude, longitude, services_offered, operating_hours, is_verified) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", vet_data)
                if cursor.rowcount > 0: inserted_vets += 1
            except Exception as e: print(f"Error inserting vet {vet_data[0]}: {e}")
        print(f"Veterinarians: Inserted {inserted_vets} new records.")
        conn.commit()

        print("\n--- Processing Disease Diagnosis (Sample) ---")
        disease_sample_data = [ # Your full list
             ('High fever, shivering, nasal discharge, cough, difficulty breathing', 'Bovine Respiratory Disease (BRD) Complex', 'Consult Vet. Antibiotics (if bacterial), anti-inflammatories, supportive care (fluids, rest), improve ventilation.', 'Medium to High', 'Common in young/stressed cattle.'),
             ('Watery diarrhea (sometimes bloody), dehydration, weakness, loss of appetite (esp. calves)', 'Scours (Calf Diarrhea)', 'Consult Vet. Fluid therapy (oral/IV electrolytes) is critical. Identify cause (viral, bacterial, protozoal) for specific treatment. Keep calf warm & clean.', 'High (in calves)', 'Multiple causes. Hygiene is key prevention.'),
             ('Sudden high fever, lameness, swelling with gas/crackling sound in large muscles (hip, shoulder)', 'Black Quarter (BQ)', 'Consult Vet Immediately. High dose Penicillin if caught extremely early. Often fatal. Vaccination is highly effective for prevention.', 'High', 'Caused by Clostridium chauvoei bacteria.'),
             ('High fever, depression, ropey saliva, nasal discharge, sudden death possible', 'Haemorrhagic Septicaemia (HS)', 'Consult Vet Immediately. Specific antibiotics (e.g., Oxytetracycline, Sulphonamides). Vaccination is crucial in endemic areas.', 'High', 'Caused by Pasteurella multocida. Common in monsoon.'),
             ('Blisters/vesicles on tongue, gums, feet (causing lameness), drooling, fever, drop in milk yield', 'Foot-and-Mouth Disease (FMD)', 'Consult Vet & Report. Highly contagious. Supportive care (soft food, antiseptic mouth/foot wash). Strict biosecurity. Vaccination provides protection.', 'High (due to economic impact)', 'Viral disease. Reportable.'),
             ('Swollen, hard, hot, painful udder quarter(s), abnormal milk (clots, watery, bloody), reduced yield, fever possible', 'Mastitis', 'Consult Vet. Intramammary antibiotics based on culture/sensitivity. Frequent milking out. Anti-inflammatories. Prevention via hygiene, proper milking.', 'Medium to High', 'Bacterial infection is common cause.'),
             ('Distended left abdomen (bloat), discomfort, difficulty breathing, kicking at belly, sudden death possible', 'Bloat (Ruminal Tympany)', 'Consult Vet. Emergency. Stomach tube to release gas. Anti-bloat medication (oils, poloxalene). For frothy bloat, trocarization may be needed. Prevent via gradual feed changes.', 'High', 'Frothy (legumes) or free gas bloat.'),
             ('Gradual weight loss despite good appetite, chronic diarrhea, reduced milk yield, bottle jaw (late stage)', "Johne's Disease (Paratuberculosis)", "Consult Vet. No cure. Test and cull positive animals to control spread. Highly infectious via manure. Long incubation period.", 'Medium (chronic, herd impact)', "Caused by Mycobacterium avium subspecies paratuberculosis."),
             ('Fever, anemia (pale gums), jaundice (yellowing), red/dark urine, weakness, tick infestation often present', 'Babesiosis / Theileriosis (Tick Fever)', 'Consult Vet. Specific anti-parasitic drugs (e.g., Diminazene, Buparvaquone). Blood transfusion if severe anemia. Tick control is essential for prevention.', 'Medium to High', 'Protozoan parasites transmitted by ticks.'),
             ('Firm, raised lumps on skin all over body, fever, swollen lymph nodes, nasal/eye discharge, drop in milk yield', 'Lumpy Skin Disease (LSD)', 'Consult Vet. Supportive care (anti-inflammatories, wound care). Antibiotics for secondary bacterial infections. Vector (insect) control helps. Vaccination available.', 'Medium to High', 'Viral disease spread by insects.')
        ]
        inserted_count_dd = 0
        for dis_data in disease_sample_data:
            try:
                cursor.execute("INSERT INTO disease_diagnosis (symptoms, detected_disease, suggested_treatment, severity, notes) VALUES (?, ?, ?, ?, ?)", dis_data)
                if cursor.rowcount > 0: inserted_count_dd +=1
            except: pass # Simple skip for brevity
        print(f"Disease Diagnosis: Inserted {inserted_count_dd} new records.")
        conn.commit()
        

        # --- This is part of your main() function in setup_cows_db.py ---
# --- It comes AFTER all CREATE TABLE statements and conn.commit() for tables ---
# --- and AFTER inserting sample users and sample user_cattle ---

        cursor = conn.cursor() # Ensure cursor is active
        print("\n--- Inserting Generic Sample Data for Logs & Reminders ---")

        # Fetch all sample farmers and their cattle to add logs for
        sample_farmers_and_cattle = []
        try:
            cursor.execute("""
                SELECT u.user_id, u.username, uc.cattle_id, uc.name
                FROM users u
                JOIN user_cattle uc ON u.user_id = uc.user_id
                WHERE u.role = 'farmer'
            """)
            for row in cursor.fetchall():
                sample_farmers_and_cattle.append({
                    "user_id": row[0],
                    "username": row[1],
                    "cattle_id": row[2],
                    "cattle_name": row[3]
                })
            if not sample_farmers_and_cattle:
                print("No sample farmer cattle found to add logs/reminders for. Ensure farmers and their cattle are created first.")
        except sqlite3.Error as e:
            print(f"Error fetching farmers and their cattle: {e}")

        total_vacc_logs_inserted = 0
        total_health_events_inserted = 0
        total_reminders_inserted = 0

        for item in sample_farmers_and_cattle:
            user_id = item["user_id"]
            username = item["username"]
            cattle_id = item["cattle_id"]
            cattle_name = item["cattle_name"] if item["cattle_name"] else f"Cattle_{cattle_id}"

            print(f"\n  Processing logs for {cattle_name} (ID: {cattle_id}) of farmer {username} (ID: {user_id})")

            # --- 1. Sample Vaccinations Log ---
            vaccinations_to_add = []
            # Generic FMD vaccination for all
            vacc_date_fmd = (date.today() - timedelta(days=random.randint(60, 180)))
            next_due_fmd = vacc_date_fmd + timedelta(days=180) # Assuming 6-month booster
            vaccinations_to_add.append(
                (cattle_id, user_id, 'FMD (Foot-and-Mouth)', vacc_date_fmd.strftime('%Y-%m-%d'),
                 next_due_fmd.strftime('%Y-%m-%d'), f'FMD{random.randint(1000,9999)}', 'Local Vet', 'Routine FMD vaccination.')
            )
            # Generic HS-BQ vaccination for all
            vacc_date_hsbq = (date.today() - timedelta(days=random.randint(30, 150)))
            next_due_hsbq = vacc_date_hsbq + timedelta(days=365) # Assuming annual
            vaccinations_to_add.append(
                (cattle_id, user_id, 'HS-BQ Combined', vacc_date_hsbq.strftime('%Y-%m-%d'),
                 next_due_hsbq.strftime('%Y-%m-%d'), f'HSBQ{random.randint(100,999)}', 'Govt. Vet Camp', 'Annual HS-BQ.')
            )
            # Conditionally add a third, more recent one for some variety
            if random.choice([True, False]):
                vacc_date_other = (date.today() - timedelta(days=random.randint(7, 25)))
                next_due_other = vacc_date_other + timedelta(days=random.choice([180, 365]))
                vaccinations_to_add.append(
                    (cattle_id, user_id, 'Deworming Internal', vacc_date_other.strftime('%Y-%m-%d'), # Using this as an example, could be another vaccine
                     next_due_other.strftime('%Y-%m-%d'), f'DEW{random.randint(100,999)}', 'Self', 'Scheduled deworming.')
                )

            current_vacc_log_ids = {} # To store log_id for creating reminders
            for v_data in vaccinations_to_add:
                try:
                    cursor.execute("""INSERT INTO vaccinations_log
                                      (cattle_id, user_id, vaccination_name, vaccination_date, next_due_date,
                                       batch_number, administered_by, notes)
                                      VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", v_data)
                    if cursor.rowcount > 0:
                        total_vacc_logs_inserted += 1
                        current_vacc_log_ids[v_data[2]] = cursor.lastrowid # Key by vaccine_name
                except sqlite3.IntegrityError: pass # Should be rare with auto PK
                except Exception as e: print(f"    Error inserting vaccination for {cattle_name}: {e}")
            if vaccinations_to_add: conn.commit()


            # --- 2. Sample Health Events Log ---
            health_events_to_add = []
            # Generic Routine Checkup
            event_date_checkup = (date.today() - timedelta(days=random.randint(20, 90)))
            next_checkup_checkup = event_date_checkup + timedelta(days=180) # 6-monthly checkup
            health_events_to_add.append(
                (cattle_id, user_id, 'Routine Health Check', event_date_checkup.strftime('%Y-%m-%d'),
                 'None specific', 'General check, found healthy', 'None', 'Dr. LocalVet',
                 next_checkup_checkup.strftime('%Y-%m-%d'), 'Good', 'Regular half-yearly checkup.')
            )
            # Conditionally add a minor illness event
            if random.choice([True, False, False]): # Make it less frequent
                event_date_illness = (date.today() - timedelta(days=random.randint(5, 50)))
                health_events_to_add.append(
                    (cattle_id, user_id, 'Minor Indigestion', event_date_illness.strftime('%Y-%m-%d'),
                     'Reduced appetite, slight bloating', 'Suspected Indigestion', 'Administered digestives, withheld concentrate', 'Self',
                     None, 'Recovered in 2 days', 'Possibly due to sudden feed change.')
                )

            current_health_event_ids = {} # To store event_id for reminders
            for he_data in health_events_to_add:
                try:
                    cursor.execute("""INSERT INTO health_events_log
                                      (cattle_id, user_id, event_type, event_date, symptoms_observed, diagnosis,
                                       treatment_administered, veterinarian_involved, next_checkup_date, outcome, notes)
                                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", he_data)
                    if cursor.rowcount > 0:
                        total_health_events_inserted += 1
                        current_health_event_ids[he_data[2]] = cursor.lastrowid # Key by event_type
                except sqlite3.IntegrityError: pass
                except Exception as e: print(f"    Error inserting health event for {cattle_name}: {e}")
            if health_events_to_add: conn.commit()

            # --- 3. Sample Health Reminders Status ---
            # Create reminders based on the next_due_date of vaccinations and next_checkup_date of health events
            reminders_for_this_animal = []

            # From Vaccinations
            for vacc_name, original_log_id in current_vacc_log_ids.items():
                # Find the corresponding vaccination entry to get its next_due_date
                vacc_entry = next((v for v in vaccinations_to_add if v[2] == vacc_name), None)
                if vacc_entry and vacc_entry[4]: # next_due_date is at index 4
                    due_date_str = vacc_entry[4]
                    reminder_desc = f"Vaccination Due: {vacc_name} for {cattle_name}"
                    reminders_for_this_animal.append(
                        (user_id, cattle_id, original_log_id, 'vaccination_due', reminder_desc, due_date_str)
                    )

            # From Health Events (for follow-up checkups)
            for event_type, original_log_id in current_health_event_ids.items():
                health_entry = next((h for h in health_events_to_add if h[2] == event_type), None)
                if health_entry and health_entry[8]: # next_checkup_date is at index 8
                    due_date_str = health_entry[8]
                    reminder_desc = f"Health Checkup Due: Follow-up for {event_type} for {cattle_name}"
                    reminders_for_this_animal.append(
                        (user_id, cattle_id, original_log_id, 'health_checkup_due', reminder_desc, due_date_str)
                    )
            
            # Add one OVERDUE reminder for demonstration for one of the cattle
            if item == sample_farmers_and_cattle[0] and len(sample_farmers_and_cattle) > 0 : # For the first cattle of the first farmer
                overdue_vacc_name = "Critical Past Vaccine"
                overdue_due_date = (date.today() - timedelta(days=random.randint(10, 40))).strftime('%Y-%m-%d')
                # Create a dummy original log for this overdue item (or link to a real past one if it exists)
                # For simplicity, let's assume original_log_id can be a placeholder if it's a manually set overdue reminder
                # Or, more correctly, we should ensure an actual past log leads to this.
                # For now, let's just create a reminder directly. If original_log_id is NOT NULL, this needs a valid ID.
                # To make it work with NOT NULL, let's pick one of the vaccination log ids if available
                dummy_original_log_id_for_overdue = list(current_vacc_log_ids.values())[0] if current_vacc_log_ids else 99999 # Placeholder if no vacc log
                
                reminders_for_this_animal.append(
                    (user_id, cattle_id, dummy_original_log_id_for_overdue, 'vaccination_due', f"OVERDUE: {overdue_vacc_name} for {cattle_name}", overdue_due_date)
                )


            for rem_data in reminders_for_this_animal:
                try:
                    cursor.execute("""INSERT INTO health_reminders_status
                                      (user_id, cattle_id, original_log_id, reminder_type, reminder_description, due_date)
                                      VALUES (?, ?, ?, ?, ?, ?)""", rem_data)
                    if cursor.rowcount > 0: total_reminders_inserted +=1
                except sqlite3.IntegrityError: pass
                except Exception as e: print(f"    Error inserting reminder for {cattle_name}: {e}")
            if reminders_for_this_animal: conn.commit()

        print(f"\nTotal Sample Vaccinations Logged: {total_vacc_logs_inserted}")
        print(f"Total Sample Health Events Logged: {total_health_events_inserted}")
        print(f"Total Sample Health Reminders Created: {total_reminders_inserted}")

        # ... (Your existing sample data insertion for other tables like cattle_for_sale, etc.) ...
        # ... (Ensure these are consistent with any schema changes and use correct user_ids if applicable) ...
        
         # --- INSIDE main() in setup_cows_db.py, after table creation and user/cattle data ---
        print("\n--- Processing Indicative Prices (Sample) ---")
        indicative_prices_data = [
         # item_type, breed_or_mach_type, category_subtype, region, low, high, source, notes, last_updated
            ('Cattle', 'Gir', 'Milking Cow (1st-2nd Lactation, Good Yield)', 'Gujarat', 60000, 95000, 'General Market Estimate 2024', 'Prices vary with exact yield, pedigree.', date.today().strftime('%Y-%m-%d')),
            ('Cattle', 'Gir', 'Heifer (Breeding Age, Good Pedigree)', 'Gujarat', 35000, 60000, 'General Market Estimate 2024', 'Dependent on lineage.', date.today().strftime('%Y-%m-%d')),
            ('Cattle', 'Sahiwal', 'Milking Cow (1st-2nd Lactation, Good Yield)', 'Punjab', 55000, 85000, 'General Market Estimate 2024', None, date.today().strftime('%Y-%m-%d')),
            ('Cattle', 'Ongole', 'Young Bull (Good Build)', 'Andhra Pradesh', 45000, 70000, 'General Market Estimate 2024', 'For draft or breeding.', date.today().strftime('%Y-%m-%d')),
            ('Cattle', 'Crossbred HF/Jersey Type', 'Milking Cow (High Yield)', 'All India', 40000, 75000, 'General Market Estimate 2024', 'Yield and health status are key.', date.today().strftime('%Y-%m-%d')),
            ('Machinery', 'Tractor', 'Used - Good Condition (35-45 HP)', 'All India', 200000, 450000, 'Market Survey Q1 2024', 'Brand and hours used affect price significantly.', date.today().strftime('%Y-%m-%d')),
            ('Machinery', 'Thresher', 'Used - Good Condition (Paddy/Wheat)', 'North India', 40000, 80000, 'Market Survey Q1 2024', None, date.today().strftime('%Y-%m-%d')),
            ('Machinery', 'Rotavator', 'Used - Good Condition (5-6 feet)', 'All India', 35000, 65000, 'Market Survey Q1 2024', None, date.today().strftime('%Y-%m-%d')),
        ]
        inserted_count_ip = 0
        for ip_data in indicative_prices_data:
            try:
                cursor.execute("""INSERT INTO indicative_prices
                        (item_type, breed_or_machinery_type, category_subtype, region,
                         price_range_low, price_range_high, data_source, notes, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", ip_data)
                if cursor.rowcount > 0: inserted_count_ip += 1
            except sqlite3.IntegrityError: pass # Should not happen with auto PK
            except Exception as e: print(f"Error inserting indicative price for {ip_data[1]}: {e}")
        print(f"Indicative Prices: Inserted {inserted_count_ip} new records.")
        conn.commit()
        
        print("\n--- Processing Impact Campaigns (Sample) ---")
        campaigns_data = [
        ('Fodder for Drought Relief', 'Provide nutritious fodder for 100 cattle in drought-affected Vidarbha region for one month.', 50000.0, 12500.0, date.today().strftime('%Y-%m-%d'), (date.today() + timedelta(days=60)).strftime('%Y-%m-%d'), 'Active', "images/fodder_campaign.jpeg", 'Fodder Support'),
        ('Urgent Vet Care Fund', 'Support emergency veterinary treatments for injured or sick stray cattle rescued by our partner shelters.', 25000.0, 5000.0, (date.today() - timedelta(days=10)).strftime('%Y-%m-%d'), (date.today() + timedelta(days=45)).strftime('%Y-%m-%d'), 'Active', "images/vet_care_campaign.jpeg", 'Veterinary Care'),
        ('Empower Dairy Farmers Training', 'Fund a 3-day training workshop on sustainable dairy practices and value addition for 20 smallholder women farmers.', 75000.0, 10000.0, (date.today() + timedelta(days=5)).strftime('%Y-%m-%d'), (date.today() + timedelta(days=90)).strftime('%Y-%m-%d'), 'Active', "images/farmer_training.jpeg", 'Farmer Training')
        ]
        inserted_campaigns = 0
        for camp_data in campaigns_data:
            try:
                cursor.execute("""INSERT INTO impact_campaigns (title, description, goal_amount, current_amount, start_date, end_date, status, image_url, category)
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", camp_data)
                if cursor.rowcount > 0: inserted_campaigns += 1
            except Exception as e: print(f"Error inserting campaign {camp_data[0]}: {e}")
        print(f"Impact Campaigns: Inserted {inserted_campaigns} new records.")
        conn.commit()

        print("\n--- Processing Adoptable Animals (Sample) ---")
        adoptable_animals_data = [
        ('Gauri', 'Cow', 'Gir', 'Female', 3.5, 'Rescued from an accident, Gauri is a gentle soul looking for a peaceful home. She loves being brushed.', 'Good, recovering well', "images/gauri_adopt.jpeg", None, 'Anand Gaushala, Gujarat', 2500.0, 'Anand Gaushala', 'contact@anandgaushala.org'),
        ('Nandi Jr.', 'Calf', 'Sahiwal', 'Male', 0.8, 'Born at our shelter, Nandi Jr. is playful and curious. He needs support for his early growth and feed.', 'Healthy, vaccinated', "images/nandi_jr_adopt.jpeg", None, 'Kamdhenu Seva Kendra, Haryana', 1500.0, 'Kamdhenu Seva Kendra', 'seva@kamdhenu.org'),
        ('Raja (Ret.)', 'Bull', 'Hallikar', 'Male', 12.0, 'Raja served local farmers faithfully for many years. He now needs a peaceful retirement and care.', 'Aged, needs joint support', "images/raja_retired_adopt.jpeg", None, 'Karuna Shelter, Karnataka', 3000.0, 'Karuna Shelter', 'info@karunashelter.in')
        ]
        inserted_adopt = 0
        for animal_data in adoptable_animals_data:
            try:
                cursor.execute("""INSERT INTO adoptable_animals (name, species, breed, sex, approx_age_years, story, health_status, image_url_1, image_url_2, location_info, monthly_sponsorship_cost, gaushala_or_org_name, contact_for_adoption)
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", animal_data)
                if cursor.rowcount > 0: inserted_adopt += 1
            except Exception as e: print(f"Error inserting adoptable animal {animal_data[0]}: {e}")
        print(f"Adoptable Animals: Inserted {inserted_adopt} new records.")
        conn.commit()



    if conn:
            conn.close()
            print("Database connection closed.")
    else:
        print("Database connection could not be established.")

if __name__ == '__main__':
    main()
    print(f"\nDatabase '{DB_FILE}' setup script finished execution.")
