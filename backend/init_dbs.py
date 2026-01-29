import sqlite3
import os
import faiss
import numpy as np

# CONFIGURACIÓN
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DATA_DIR, "galeria.db")
FAISS_PATH = os.path.join(DATA_DIR, "embeddings.index")
DIMENSION_EMBEDDING = 512 # ArcFace usa 512 dimensiones

# --- ESQUEMA SQL ---
SQL_SCHEMA = """
-- Habilitar Foreign Keys
PRAGMA foreign_keys = ON;

-- 1. TABLA FOTOS
CREATE TABLE IF NOT EXISTS fotos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ruta_archivo TEXT NOT NULL UNIQUE,
    hash_md5 TEXT UNIQUE,              -- Para detectar duplicados al importar
    
    -- Metadatos básicos
    fecha_creacion TIMESTAMP,          -- EXIF Original
    fecha_importacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ancho INTEGER,
    alto INTEGER,
    tamanio_bytes INTEGER,
    
    -- Estado del procesamiento
    procesada_facial BOOLEAN DEFAULT 0
);

-- Índices para búsqueda rápida
CREATE INDEX IF NOT EXISTS idx_fotos_hash ON fotos(hash_md5);
CREATE INDEX IF NOT EXISTS idx_fotos_fecha ON fotos(fecha_creacion);

-- 2. TABLA PERSONAS (Identidades)
CREATE TABLE IF NOT EXISTS personas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT NOT NULL DEFAULT 'Desconocido',
    es_conocida BOOLEAN DEFAULT 0,
    foto_avatar_id INTEGER,
    
    etiqueta_id INTEGER, 
    
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (foto_avatar_id) REFERENCES fotos(id) ON DELETE SET NULL,
    FOREIGN KEY (etiqueta_id) REFERENCES etiquetas(id) ON DELETE SET NULL
);

-- 3. TABLA ROSTROS_DETECTADOS (El puente SQL <-> FAISS)
CREATE TABLE IF NOT EXISTS rostros_detectados (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- ESTE ID SE USARÁ EN FAISS
    foto_id INTEGER NOT NULL,
    persona_id INTEGER,                -- A quién pertenece (puede ser NULL al inicio)
    
    -- Dónde está la cara en la foto
    bbox_x INTEGER, bbox_y INTEGER, 
    bbox_w INTEGER, bbox_h INTEGER,
    score_confianza REAL,

    -- Respaldo del vector (BLOB) por seguridad
    embedding_blob BLOB, 

    FOREIGN KEY (foto_id) REFERENCES fotos(id) ON DELETE CASCADE,
    FOREIGN KEY (persona_id) REFERENCES personas(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_rostros_persona ON rostros_detectados(persona_id);

-- 4. TABLA ETIQUETAS
CREATE TABLE IF NOT EXISTS etiquetas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    texto TEXT NOT NULL UNIQUE,
    tipo TEXT DEFAULT 'general',       -- 'ubicacion', 'evento', 'usuario'
    color TEXT
);

-- 5. RELACIÓN FOTOS <-> ETIQUETAS
CREATE TABLE IF NOT EXISTS fotos_etiquetas (
    foto_id INTEGER NOT NULL,
    etiqueta_id INTEGER NOT NULL,
    asignacion_manual BOOLEAN DEFAULT 1,
    
    PRIMARY KEY (foto_id, etiqueta_id),
    FOREIGN KEY (foto_id) REFERENCES fotos(id) ON DELETE CASCADE,
    FOREIGN KEY (etiqueta_id) REFERENCES etiquetas(id) ON DELETE CASCADE
);
"""

def init_directorios():
    """Crea la carpeta de datos si no existe"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 Directorio creado: {DATA_DIR}")
    else:
        print(f"📁 Directorio existente: {DATA_DIR}")

def init_sql():
    """Inicializa la base de datos SQLite"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.executescript(SQL_SCHEMA)
        conn.commit()
        conn.close()
        print(f"✅ Base de datos SQL inicializada en: {DB_PATH}")
    except Exception as e:
        print(f"❌ Error inicializando SQL: {e}")

def init_faiss():
    """Inicializa el índice vectorial FAISS"""
    if os.path.exists(FAISS_PATH):
        print(f"✅ Índice FAISS ya existe en: {FAISS_PATH}")
        # Opcional: Cargar para verificar integridad
        # index = faiss.read_index(FAISS_PATH)
        return

    print("⚙️ Creando nuevo índice FAISS...")
    
    index_base = faiss.IndexFlatIP(DIMENSION_EMBEDDING)
    index_con_ids = faiss.IndexIDMap(index_base)

    faiss.write_index(index_con_ids, FAISS_PATH)
    print(f"✅ Índice FAISS creado y guardado en: {FAISS_PATH}")

if __name__ == "__main__":
    print("🚀 Iniciando configuración de las bases de datos...")
    init_directorios()
    init_sql()
    init_faiss()
    print("✨ Todo listo. Puedes empezar a construir la API.")