import sqlite3
import pandas as pd

def check_database():
    db_path = 'data/outputs/events.db'
    
    # Conecta ao banco
    with sqlite3.connect(db_path) as conn:
        print("\n=== 1. CONTAGEM POR CLASSE DE VEÍCULO ===")
        query1 = """
        SELECT vehicle_class, COUNT(*) as total 
        FROM vehicle_events 
        GROUP BY vehicle_class 
        ORDER BY total DESC;
        """
        # Usando pandas apenas para imprimir a tabela bem formatada no terminal
        df1 = pd.read_sql_query(query1, conn)
        print(df1.to_string(index=False))

        print("\n=== 2. PLACAS DETECTADAS ===")
        query2 = """
        SELECT track_id, vehicle_class, plate_text, plate_confidence, frame_number
        FROM vehicle_events 
        WHERE plate_text IS NOT NULL;
        """
        df2 = pd.read_sql_query(query2, conn)
        if df2.empty:
            print("Nenhuma placa registrada ainda.")
        else:
            print(df2.to_string(index=False))
        print("\n")

if __name__ == "__main__":
    check_database()