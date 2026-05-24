import sqlite3
import datetime
import os

DB_PATH = 'output/parking.db'

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS parking_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            time_in DATETIME NOT NULL,
            time_out DATETIME,
            status TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def log_vehicle(plate_number: str):
    """
    Log a vehicle. 
    If the vehicle is already IN the parking lot, mark as OUT.
    Otherwise, mark as IN.
    Returns: (status, time) string indicating 'IN' or 'OUT' and the current time.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Kiểm tra xem xe này có đang ở trong bãi không (status = 'IN')
    cursor.execute('''
        SELECT id FROM parking_history 
        WHERE plate_number = ? AND status = 'IN' 
        ORDER BY time_in DESC LIMIT 1
    ''', (plate_number,))
    
    record = cursor.fetchone()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if record:
        # Xe đang trong bãi -> Đánh dấu đi ra (CHECK-OUT)
        record_id = record[0]
        cursor.execute('''
            UPDATE parking_history 
            SET time_out = ?, status = 'OUT' 
            WHERE id = ?
        ''', (current_time, record_id))
        conn.commit()
        conn.close()
        return "OUT", current_time
    else:
        # Xe chưa có trong bãi -> Đánh dấu đi vào (CHECK-IN)
        cursor.execute('''
            INSERT INTO parking_history (plate_number, time_in, status) 
            VALUES (?, ?, 'IN')
        ''', (plate_number, current_time))
        conn.commit()
        conn.close()
        return "IN", current_time
