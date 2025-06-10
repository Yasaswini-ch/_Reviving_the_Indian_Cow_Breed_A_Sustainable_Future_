import sqlite3
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin SDK
cred = credentials.Certificate(r"C:\Users\chebo\Kamdhenu_App-main\indian-cow-breed-qodc-firebase-adminsdk-fbsvc-710313aa8d.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Your existing dictionary of known tables and their primary keys
primary_keys = {
    "cattle_breeds": "breed_id",
    "government_schemes": "scheme_id",
    "eco_practices": "practice_id",
    "disease_diagnosis": "report_id",
    "price_trends": "trend_id",
    "image_analysis": "image_id",
    "user_queries": "query_id",
    "saved_alerts": "alert_id",
    "user_cattle": "cattle_id",
    "users": "user_id",
    "calf_rearing_log": "calf_log_id",
    "vaccination_records": "vaccination_id",
    "vaccinations_log": "log_id",
    "health_records": "health_record_id",
    "health_events_log": "event_id",
    "health_reminders_status": "reminder_id",
    "cattle_for_sale": "listing_id",
    "machinery_listings": "machinery_id",
    "breeding_pairs": "pair_id",
    "veterinarians": "vet_id",
    "offspring_data": "offspring_data_id",
    "user_saved_listings": "saved_id",
    "feedstuffs": "feedstuff_id",
    "milk_log": "log_id",
    "indicative_prices":"price_guide_id",
    "impact_campaigns":"campaign_id",
    "donations_log":"log_id",
    "adoptions_log":"adoption_id",
    "adoptable_animals":"adoptable_animal_id"
}

def get_all_tables(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    return set(row[0] for row in cursor.fetchall())

def get_table_data(conn, table):
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table}")
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    return columns, rows

def migrate_table(conn, table, primary_key):
    columns, rows = get_table_data(conn, table)
    for row in rows:
        data = dict(zip(columns, row))
        doc_id = str(data[primary_key])  # Document ID as primary key value
        doc_ref = db.collection(table).document(doc_id)
        doc_ref.set(data)
    print(f"Synced {len(rows)} records for table '{table}'")

def main():
    conn = sqlite3.connect('Cows.db')
    tables_in_db = get_all_tables(conn)

    # Check for new tables that have no primary key defined
    unknown_tables = tables_in_db - set(primary_keys.keys())
    if unknown_tables:
        print("⚠️ WARNING: The following tables have no primary key defined in your script:")
        for tbl in unknown_tables:
            print(f" - {tbl}")
        print("Please add their primary keys in the 'primary_keys' dictionary before proceeding.")
        # Optionally exit or raise error
        return

    # Proceed with migration of known tables
    for table in tables_in_db:
        pk = primary_keys.get(table)
        if not pk:
            # This should never happen because of the check above, but safe check anyway
            print(f"Skipping table '{table}' as it has no primary key defined.")
            continue
        print(f"Migrating table '{table}' with primary key '{pk}'...")
        migrate_table(conn, table, pk)

    print("✅ Migration completed successfully.")

if __name__ == "__main__":
    main()
