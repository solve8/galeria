import json
import os
import sys
import subprocess
import faiss
import numpy as np
import sqlite3
from backend.db_manager import DatabaseManager

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

class ProcesadorDeFotos:
    '''Se encarga de:
        - calcular los embeddings de las caras de una foto
        - registrar nuevas personas
        - añadir las etiquetas'''
    def __init__(self, id_foto):
        self.db = DatabaseManager()
        self.id_foto = id_foto
        self.ruta_foto = self.db.obtener_ruta_foto(id_foto)

        if not self.ruta_foto:
            raise ValueError(f"❌ Error: No existe la foto con ID {id_foto} en la base de datos.")
        
        if not os.path.exists(self.ruta_foto):
            raise FileNotFoundError(f"❌ Error: El archivo físico no existe: {self.ruta_foto}")

    def obtener_caras(self, ruta_foto_nueva):
        '''Obtiene las caras de una foto.'''
        print(f"🚀 Iniciando sistema con {MODELO} y detector {DETECTOR}...")
        print(f"😃 Obteniendo caras: {ruta_foto_nueva}...")
        
        try:
            caras = DeepFace.represent(
                img_path=ruta_foto_nueva, 
                model_name=MODELO,
                detector_backend=DETECTOR,
                enforce_detection=True  
            )
            
            if not caras: # si por alguna razón la lista llega vacía sin lanzar excepción
                raise ValueError("Lista de caras vacía")

            print(f"   -> Detectadas {len(caras)} caras.")
            return caras

        except ValueError:
            print(f"⚠️ Aviso: No se detectó ninguna cara en {ruta_foto_nueva}.")
            return []
            
        except Exception as e:
            print(f"❌ Error inesperado procesando la imagen: {e}")
            return []

    def procesando_caras(self, caras):
        '''Dada una cara:
            1. busca coincidencias en la base de datos FAISS:
                a. Si no encuentra coincidencia -> añadir a FAISS y crear persona
            2. añadir la etiqueta de la persona a la foto'''
            
        ids_encontrados = []
        
        for i, cara in enumerate(caras):
            embedding = np.array(cara["embedding"], dtype='float32') # convertilo en un bloque de memoria contiguo
            
            print(f"\n   🔍 Analizando cara {i+1}...")
            
            id_persona, distancia = self.db.identificar_persona_por_vector(embedding)
            
            if id_persona:
                ids_encontrados.append(id_persona)
            else:
                nuevo_id = self.db.registrar_nueva_persona(
                    foto_id=15,
                    embedding=embedding,
                    area=cara["facial_area"],
                    nombre="Desconocido"
                )
                print(f"      🆕 DESCONOCIDO: Registrado como ID {nuevo_id}")
                print(f"      (Distancia más cercana: {distancia:.4f})")
                ids_encontrados.append(nuevo_id)
        
        print(f"\n✅ Procesamiento completado. IDs encontrados: {ids_encontrados}")
        return ids_encontrados

    def añadir_etiquetas(self, ids_encontrados):
        """
        Recorre los IDs de personas detectadas, busca sus etiquetas correspondientes
        y las asigna a la foto actual usando el DatabaseManager.
        """
        if not ids_encontrados:
            return

        print(f"🏷️ Asignando etiquetas para {len(ids_encontrados)} personas...")

        for persona_id in ids_encontrados:
            # 1. Averiguar cuál es la etiqueta de esta persona
            etiqueta_id = self.db.obtener_etiqueta_id_de_persona(persona_id)
            
            if etiqueta_id:
                # 2. Usar el método genérico del Manager para asignar
                # Asumimos que tienes el ID de la foto actual guardado en self.foto_id_actual
                # Si no, pásalo como argumento a esta función.
                self.db.asignar_etiqueta(
                    foto_id=self.foto_id_actual, 
                    etiqueta_id=etiqueta_id, 
                    manual=False # Es automático
                )
                print(f"   -> Etiqueta (ID {etiqueta_id}) asignada a foto {self.foto_id_actual}")
            else:
                print(f"   ⚠️ La persona ID {persona_id} no tiene etiqueta vinculada.")

    def procesar_foto(self, ruta_foto):
        """
        Método maestro que ejecuta todo el flujo.
        """
        print(f"📸 --- PROCESANDO FOTO ID: {self.id_foto} ---")
        
        caras = self.obtener_caras()
        
        if not caras:
            print("⏹️ Fin del procesamiento (sin caras).")
            self.db.marcar_foto_como_procesada(self.id_foto)
            return

        ids_personas = self.procesando_caras(caras)
        
        self.añadir_etiquetas(ids_personas)
        
        self.db.marcar_foto_como_procesada(self.id_foto)
        
        print(f"✅ FOTO {self.id_foto} COMPLETADA.\n")

        '''Ejemplo de uso:
        # Supongamos que acabas de importar una foto y tienes su ID (ej. 100)
        try:
            procesador = ProcesadorDeFotos(100) # Él solo busca la ruta y se prepara
            procesador.procesar()               # Hace toda la magia
        except Exception as e:
            print(f"Error: {e}")'''