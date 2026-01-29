import json
import os
import sys
import subprocess
import faiss
import numpy as np
import sqlite3

# Instalación de dependencias
try:
    from deepface import DeepFace
except ImportError:
    print("Instalando librerías necesarias...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "deepface", "tf-keras", "retina-face", "faiss-cpu"])
    from deepface import DeepFace

# CONFIGURACIÓN
UMBRAL_COINCIDENCIA = 0.55 
MODELO = "ArcFace"
DETECTOR = "retinaface"
DIMENSION_EMBEDDING = 512  # ArcFace genera embeddings de 512 dimensiones

class SistemaReconocimiento:
    def __init__(self, db_sql="personas.db", archivo_faiss="embeddings.index"):
        self.db_sql = db_sql
        self.archivo_faiss = archivo_faiss
        
        self._init_sql()
        self._init_faiss()
    
    def _init_sql(self):
        """Crea la tabla si no existe"""
        conn = sqlite3.connect(self.db_sql)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS personas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT,
                fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                foto_original TEXT,
                modelo_usado TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def _init_faiss(self):
        """Carga o crea índice FAISS"""
        if os.path.exists(self.archivo_faiss):
            self.index = faiss.read_index(self.archivo_faiss)
            print(f"📚 FAISS cargado: {self.index.ntotal} embeddings")
        else:
            self.index = faiss.IndexFlatIP(DIMENSION_EMBEDDING)
            print("✨ Nuevo índice FAISS creado")
    
    def _guardar_faiss(self):
        """Guarda el índice FAISS a disco"""
        faiss.write_index(self.index, self.archivo_faiss)
    
    def _get_proximo_id(self):
        """Obtiene el siguiente ID disponible"""
        conn = sqlite3.connect(self.db_sql)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(id) FROM personas")
        resultado = cursor.fetchone()[0]
        conn.close()
        return (resultado or 0) + 1
    
    def registrar_persona(self, embedding, nombre=None, foto_original=None):
        """Registra una nueva persona en SQL y FAISS"""
        # 1. Insertar en SQL
        conn = sqlite3.connect(self.db_sql)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO personas (nombre, foto_original, modelo_usado) VALUES (?, ?, ?)",
            (nombre, foto_original, MODELO)
        )
        nuevo_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # 2. Agregar a FAISS (normalizar para similitud coseno)
        embedding_normalizado = embedding / np.linalg.norm(embedding)
        self.index.add(embedding_normalizado.reshape(1, -1))
        self._guardar_faiss()
        
        return nuevo_id
    
    def buscar_coincidencia(self, embedding, k=1):
        """
        Busca la coincidencia más cercana
        Retorna: (id_persona, similitud) o (None, None)
        """
        if self.index.ntotal == 0:
            return None, None
        
        # Normalizar query
        embedding_normalizado = embedding / np.linalg.norm(embedding)
        
        # Búsqueda (devuelve similitud, no distancia)
        similitudes, indices = self.index.search(embedding_normalizado.reshape(1, -1), k)
        
        mejor_similitud = similitudes[0][0]
        mejor_idx = indices[0][0]
        
        # Convertir similitud a distancia coseno: distancia = 1 - similitud
        distancia = 1 - mejor_similitud
        
        if distancia < UMBRAL_COINCIDENCIA:
            # El ID en FAISS corresponde a la posición, que equivale al ID de SQL
            # (porque insertamos en orden)
            id_persona = mejor_idx + 1  # FAISS indexa desde 0, SQL desde 1
            return id_persona, distancia
        
        return None, distancia
    
    def obtener_info_persona(self, id_persona):
        """Obtiene metadatos de una persona"""
        conn = sqlite3.connect(self.db_sql)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM personas WHERE id = ?", (id_persona,))
        resultado = cursor.fetchone()
        conn.close()
        
        if resultado:
            return {
                "id": resultado[0],
                "nombre": resultado[1],
                "fecha_registro": resultado[2],
                "foto_original": resultado[3],
                "modelo_usado": resultado[4]
            }
        return None

    def procesar_foto(self,ruta_foto_nueva):
        print(f"🚀 Iniciando sistema con {MODELO} y detector {DETECTOR}...")
        
        sistema = SistemaReconocimiento()
        
        # Extraer embeddings de la foto nueva
        print(f"📸 Procesando imagen: {ruta_foto_nueva}...")
        try:
            caras = DeepFace.represent(
                img_path=ruta_foto_nueva, 
                model_name=MODELO,
                detector_backend=DETECTOR,
                enforce_detection=False
            )
        except Exception as e:
            print(f"❌ Error procesando la imagen: {e}")
            return

        print(f"   -> Detectadas {len(caras)} caras.")
        
        ids_encontrados = []
        
        for i, cara in enumerate(caras):
            embedding = np.array(cara["embedding"], dtype='float32')
            
            print(f"\n   🔍 Analizando cara {i+1}...")
            
            # Buscar coincidencia
            id_persona, distancia = sistema.buscar_coincidencia(embedding)
            
            if id_persona:
                info = sistema.obtener_info_persona(id_persona)
                print(f"      ✅ IDENTIFICADO: {info['nombre'] or f'ID {id_persona}'}")
                print(f"      (Distancia: {distancia:.4f} / Umbral: {UMBRAL_COINCIDENCIA})")
                ids_encontrados.append(id_persona)
            else:
                # Registrar nueva persona
                nuevo_id = sistema.registrar_persona(
                    embedding, 
                    nombre=f"Persona_{sistema._get_proximo_id()}",
                    foto_original=ruta_foto_nueva
                )
                print(f"      🆕 DESCONOCIDO: Registrado como ID {nuevo_id}")
                print(f"      (Distancia más cercana: {distancia:.4f})")
                ids_encontrados.append(nuevo_id)
        
        print(f"\n✅ Procesamiento completado. IDs encontrados: {ids_encontrados}")

if __name__ == "__main__":
    sistema = SistemaReconocimiento()
    sistema.procesar_foto("ambos.jpg")