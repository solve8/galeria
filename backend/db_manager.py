import sqlite3
import faiss
import numpy as np
import os
import json
from init_dbs import DB_PATH, FAISS_PATH, DIMENSION_EMBEDDING


class DatabaseManager:
    def __init__(self, db_path=DB_PATH, faiss_path=FAISS_PATH, dimension=DIMENSION_EMBEDDING):
        self.db_path = db_path
        self.faiss_path = faiss_path
        self.dimension = dimension
        self.conn = None
        self.index = None
        
        self.conectar()

    def conectar(self):
        """Abre conexión a SQL y carga FAISS en RAM"""
        # 1. SQL: check_same_thread=False es vital para FastAPI (multihilo)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row # Para acceder a columnas por nombre
        
        # 2. FAISS
        if os.path.exists(self.faiss_path):
            self.index = faiss.read_index(self.faiss_path)
        else:
            # Si no existe, creamos uno nuevo vacío con IDMap
            index_base = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIDMap(index_base)

    def guardar_cambios_faiss(self):
        """FAISS vive en RAM. Hay que guardar en disco explícitamente."""
        faiss.write_index(self.index, self.faiss_path)

    # ==========================================
    # GESTIÓN DE FOTOS
    # ==========================================
    def registrar_foto(self, ruta, hash_md5, metadatos):
        """
        Inserta foto en SQL. Si ya existe (por hash), devuelve su ID existente.
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                """INSERT INTO fotos (ruta_archivo, hash_md5, fecha_creacion, ancho, alto) 
                   VALUES (?, ?, ?, ?, ?)""",
                (ruta, hash_md5, metadatos.get('fecha'), metadatos.get('ancho'), metadatos.get('alto'))
            )
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Si el hash ya existe, devolvemos el ID de la foto original
            cursor.execute("SELECT id FROM fotos WHERE hash_md5 = ?", (hash_md5,))
            return cursor.fetchone()['id']

    # ==========================================
    # GESTIÓN DE ROSTROS (HÍBRIDO SQL + FAISS)
    # ==========================================
    def guardar_rostros_detectados(self, foto_id, lista_rostros_deepface):
        """
        1. Guarda metadatos en SQL -> Obtenemos IDs
        2. Guarda vectores en FAISS usando ESOS IDs
        """
        cursor = self.conn.cursor()
        ids_generados = []
        vectores_para_faiss = []
        ids_para_faiss = []

        for rostro in lista_rostros_deepface:
            embedding = rostro['embedding']
            area = rostro['facial_area']
            confianza = rostro['face_confidence']

            # A. Insertar en SQL para obtener un ID único
            cursor.execute(
                """INSERT INTO rostros_detectados 
                   (foto_id, bbox_x, bbox_y, bbox_w, bbox_h, score_confianza) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (foto_id, area['x'], area['y'], area['w'], area['h'], confianza)
            )
            nuevo_id_sql = cursor.lastrowid
            
            # B. Preparar datos para FAISS
            ids_generados.append(nuevo_id_sql)
            vectores_para_faiss.append(embedding)
            ids_para_faiss.append(nuevo_id_sql)

        self.conn.commit()

        # C. Insertar en FAISS (Batch operation es más rápida)
        if vectores_para_faiss:
            # Convertir a numpy float32 y normalizar (para similitud coseno)
            matriz_vectores = np.array(vectores_para_faiss).astype('float32')
            faiss.normalize_L2(matriz_vectores) 
            
            array_ids = np.array(ids_para_faiss).astype('int64')
            
            # MAGIA: Insertamos vector + ID específico
            self.index.add_with_ids(matriz_vectores, array_ids)
            self.guardar_cambios_faiss()

        return ids_generados

    # ==========================================
    # BÚSQUEDA (HÍBRIDO FAISS -> SQL)
    # ==========================================
    def buscar_rostros_similares(self, vector_query, limite=5, umbral_coseno=0.55):
        """
        Busca en FAISS y luego enriquece con datos de SQL
        """
        # 1. Preparar vector
        vector_np = np.array([vector_query]).astype('float32')
        faiss.normalize_L2(vector_np)

        # 2. Búsqueda en FAISS
        # distancias = similitud (coseno). Mayor es mejor (cerca de 1.0)
        similitudes, ids_rostros = self.index.search(vector_np, k=limite)
        
        resultados = []
        cursor = self.conn.cursor()

        # 3. Cruzar datos
        for i, id_rostro in enumerate(ids_rostros[0]):
            similitud = similitudes[0][i]
            distancia = 1 - similitud # Convertimos a distancia para comparar con tu umbral

            # Filtro: Si el ID es -1 (no encontrado) o la distancia es muy grande
            if id_rostro == -1 or distancia > umbral_coseno:
                continue

            # Recuperar info de SQL
            cursor.execute("""
                SELECT r.id, p.nombre, f.ruta_archivo 
                FROM rostros_detectados r
                LEFT JOIN personas p ON r.persona_id = p.id
                JOIN fotos f ON r.foto_id = f.id
                WHERE r.id = ?
            """, (int(id_rostro),))
            
            data = cursor.fetchone()
            if data:
                resultados.append({
                    "id_rostro": data['id'],
                    "nombre_persona": data['nombre'],
                    "ruta_foto": data['ruta_archivo'],
                    "similitud": float(similitud),
                    "distancia": float(distancia)
                })

        return resultados

    # En backend/database/db_manager.py

    def identificar_persona_por_vector(self, embedding, umbral=0.55):
        """
        Recibe un embedding y busca en FAISS.
        Si encuentra coincidencia, consulta SQL para saber a qué Persona pertenece ese rostro y devuelve el id.
        """
        if self.index.ntotal == 0:  # si no hay caras guardadas
            return None, 1.0 # ID None, Distancia Máxima

        # 1. Preparar vector para FAISS (float32 y normalizado)
        vector_np = np.array([embedding]).astype('float32')
        faiss.normalize_L2(vector_np)
        
        # 2. Buscar el vecino más cercano (k=1)
        similitudes, ids_rostros = self.index.search(vector_np, k=1)
        
        mejor_similitud = similitudes[0][0]
        id_rostro_encontrado = ids_rostros[0][0] # Este es el ID de la tabla rostros_detectados
        
        # 3. Calcular distancia coseno
        distancia = 1.0 - mejor_similitud
        
        # 4. Validar umbral
        if distancia < umbral and id_rostro_encontrado != -1:
            # ¡MATCH EN FAISS!
            # Ahora preguntamos a SQL: "¿A qué persona pertenece este rostro ID X?"
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT persona_id FROM rostros_detectados WHERE id = ?", 
                (int(id_rostro_encontrado),)
            )
            resultado = cursor.fetchone()
            
            if resultado and resultado['persona_id']:
                return resultado['persona_id'], distancia
        
        # Si no supera el umbral o no se encuentra persona asociada
        return None, distancia

    def crear_o_recuperar_etiqueta(self, texto, tipo='persona', color='#3498db'):
        """
        Crea una etiqueta si no existe, o devuelve el ID de la existente.
        Ideal para gestionar nombres de personas como etiquetas.
        """
        cursor = self.conn.cursor()
        try:
            # Intentamos insertar
            cursor.execute(
                "INSERT INTO etiquetas (texto, tipo, color) VALUES (?, ?, ?)", 
                (texto, tipo, color)
            )
            self.conn.commit()
            return cursor.lastrowid
            
        except sqlite3.IntegrityError:
            # Si falla porque el texto ya existe (UNIQUE constraint), recuperamos su ID
            # Hacemos rollback parcial para limpiar el estado de error de la conexión
            self.conn.rollback() 
            
            cursor.execute("SELECT id FROM etiquetas WHERE texto = ?", (texto,))
            resultado = cursor.fetchone()
            if resultado:
                return resultado[0] # Devolvemos el ID existente
            return None

    def registrar_nueva_persona(self, foto_id, embedding, area, nombre="Persona Nueva"):
        """
        Registra una nueva identidad en el sistema.
        
        Args:
            foto_id (int): ID de la foto en la tabla 'fotos'.
            embedding (list/np.array): El vector de 512 números.
            area (dict/list): Coordenadas {'x':..., 'y':...} o lista.
            nombre (str): Nombre para asignar.
            
        Returns:
            int: El ID autogenerado de la nueva persona (persona_id).
        """
        cursor = self.conn.cursor()
        
        try:
            # 1. SQL: Crear la Persona
            cursor.execute(
                "INSERT INTO personas (nombre, es_conocida) VALUES (?, 0)", 
                (nombre,)
            )
            nuevo_persona_id = cursor.lastrowid
            
            # 2. GENERAR NOMBRE AUTOMÁTICO (Si no venía uno)
            if not nombre:
                nombre_final = f"Desconocido {nuevo_persona_id}"
            else:
                nombre_final = nombre

            # 3. CREAR ETIQUETA Y ASIGNAR
            etiqueta_id = self.crear_o_recuperar_etiqueta(texto=nombre_final, tipo='persona')
            
            cursor.execute(
                "UPDATE personas SET nombre = ?, etiqueta_id = ? WHERE id = ?",
                (nombre_final, etiqueta_id, nuevo_persona_id)
            )

            # 2. SQL: Guardar el Rostro detectado
            # Convertimos el embedding a bytes (BLOB) para respaldo en SQL
            embedding_blob = np.array(embedding, dtype='float32').tobytes()
            
            # Normalizamos datos del área (por si vienen como dict o lista)
            x = area['x'] if isinstance(area, dict) else area[0]
            y = area['y'] if isinstance(area, dict) else area[1]
            w = area['w'] if isinstance(area, dict) else area[2]
            h = area['h'] if isinstance(area, dict) else area[3]
            
            cursor.execute(
                """INSERT INTO rostros_detectados 
                   (foto_id, persona_id, bbox_x, bbox_y, bbox_w, bbox_h, embedding_blob) 
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (foto_id, nuevo_persona_id, x, y, w, h, embedding_blob)
            )
            nuevo_rostro_id = cursor.lastrowid # Este ID es vital para FAISS
            
            self.conn.commit()

            # 3. FAISS: Indexar el vector
            vector_np = np.array([embedding]).astype('float32')
            faiss.normalize_L2(vector_np)
            
            # Usamos el ID del ROSTRO para FAISS (no el de persona), 
            # así si borras este rostro, no borras a la persona entera.
            id_faiss = np.array([nuevo_rostro_id]).astype('int64')
            
            self.index.add_with_ids(vector_np, id_faiss)
            self.guardar_cambios_faiss()

            print(f"✅ Persona registrada: {nombre} (ID: {nuevo_persona_id})")
            return nuevo_persona_id

        except Exception as e:
            self.conn.rollback()
            print(f"❌ Error al registrar persona: {e}")
            return None

    def renombrar_persona(self, persona_id, nuevo_nombre):
        """
        Cambia el nombre de una persona y actualiza su etiqueta automáticamente.
        Ej: De "Persona 55" a "Juan".
        """
        cursor = self.conn.cursor()
        try:
            # 1. Obtener el nombre antiguo para buscar su etiqueta
            cursor.execute("SELECT nombre FROM personas WHERE id = ?", (persona_id,))
            res = cursor.fetchone()
            if not res:
                return False
            nombre_antiguo = res['nombre']

            print(f"🔄 Renombrando '{nombre_antiguo}' a '{nuevo_nombre}'...")

            # 2. Actualizar tabla PERSONAS
            cursor.execute(
                "UPDATE personas SET nombre = ?, es_conocida = 1 WHERE id = ?", 
                (nuevo_nombre, persona_id)
            )

            # 3. Actualizar tabla ETIQUETAS
            # Buscamos la etiqueta que tenía el nombre antiguo y le ponemos el nuevo
            cursor.execute(
                "UPDATE etiquetas SET texto = ? WHERE texto = ? AND tipo = 'persona'",
                (nuevo_nombre, nombre_antiguo)
            )
            
            # (Opcional) Si la etiqueta nueva YA existía (ej. fusionar dos personas), 
            # SQLite daría error de UNIQUE en el paso 3. 
            # Eso requeriría lógica de fusión más compleja (merge), 
            # pero para empezar, un renombrado simple basta.

            self.conn.commit()
            return True

        except sqlite3.IntegrityError:
            print("⚠️ El nombre ya existe como etiqueta. (Aquí podrías implementar fusión de personas)")
            self.conn.rollback()
            return False
        except Exception as e:
            print(f"❌ Error al renombrar: {e}")
            self.conn.rollback()
            return False

    def asignar_etiqueta(self, foto_id, etiqueta_id, manual=False):
        """
        Método GENÉRICO para pegar una etiqueta a una foto.
        Sirve para personas, eventos, lugares, etc.
        """
        cursor = self.conn.cursor()
        try:
            # INSERT OR IGNORE es vital para no fallar si ya estaba etiquetada
            cursor.execute(
                """INSERT OR IGNORE INTO fotos_etiquetas 
                   (foto_id, etiqueta_id, asignacion_manual) VALUES (?, ?, ?)""",
                (foto_id, etiqueta_id, 1 if manual else 0)
            )
            self.conn.commit()
            return True
        except Exception as e:
            print(f"❌ Error asignando etiqueta: {e}")
            return False

    def obtener_etiqueta_id_de_persona(self, persona_id):
        """
        Devuelve el ID de la etiqueta asociada a una persona.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT etiqueta_id FROM personas WHERE id = ?", (persona_id,))
        resultado = cursor.fetchone()
        if resultado and resultado[0]:
            return resultado[0]
        return None

    def obtener_ruta_foto(self, foto_id):
        """Devuelve la ruta de archivo de una foto dado su ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT ruta_archivo FROM fotos WHERE id = ?", (foto_id,))
        resultado = cursor.fetchone()
        if resultado:
            return resultado['ruta_archivo'] # Asumiendo row_factory = sqlite3.Row
            # Si no usas row_factory, sería resultado[0]
        return None

    def marcar_foto_como_procesada(self, foto_id):
        """
        Marca la foto como procesada facialmente en la base de datos
        para evitar re-escanearla en el futuro.
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "UPDATE fotos SET procesada_facial = 1 WHERE id = ?", 
                (foto_id,)
            )
            self.conn.commit()
            return True
        except Exception as e:
            print(f"❌ Error al marcar foto {foto_id} como procesada: {e}")
            return False

    def cerrar(self):
        if self.conn:
            self.conn.close()
        # Asegurar guardado final
        self.guardar_cambios_faiss()