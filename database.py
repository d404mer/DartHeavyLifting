"""
Модуль для работы с PostgreSQL базой данных
Создает базу данных DartServer и таблицы при запуске приложения
"""

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Менеджер для работы с PostgreSQL базой данных"""
    
    def __init__(self, host='localhost', port=5432, user='postgres', password='postgres', dbname='postgres'):
        """
        Инициализация менеджера БД
        
        Args:
            host: Хост PostgreSQL сервера
            port: Порт PostgreSQL сервера
            user: Имя пользователя
            password: Пароль
            dbname: Имя базы данных для подключения (обычно postgres для создания новой БД)
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.dbname = dbname
        self.target_dbname = 'DartServer'
        self.conn = None
        
    def connect_to_postgres(self):
        """Подключение к PostgreSQL серверу (к базе postgres)"""
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                dbname=self.dbname,
                connect_timeout=5
            )
            self.conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            logger.info(f"✅ Подключение к PostgreSQL серверу установлено ({self.host}:{self.port})")
            return True
        except psycopg2.OperationalError as e:
            error_msg = str(e).strip()
            logger.error(f"❌ Ошибка подключения к PostgreSQL: {error_msg}")
            if "password" in error_msg.lower() or "authentication" in error_msg.lower():
                print(f"\n⚠️  Проверьте параметры подключения к PostgreSQL:")
                print(f"   - Пользователь: {self.user}")
                print(f"   - Пароль: {'***' if self.password else 'не указан'}")
                print(f"   - Хост: {self.host}:{self.port}")
            elif "could not connect" in error_msg.lower() or "connection refused" in error_msg.lower():
                print(f"\n⚠️  PostgreSQL сервер недоступен на {self.host}:{self.port}")
                print(f"   Убедитесь, что PostgreSQL запущен и доступен.")
            return False
        except Exception as e:
            error_msg = str(e).strip()
            logger.error(f"❌ Неожиданная ошибка подключения к PostgreSQL: {error_msg}")
            print(f"\n❌ Ошибка: {error_msg}")
            return False
    
    def create_database(self):
        """Создание базы данных DartServer если её нет"""
        if not self.conn:
            if not self.connect_to_postgres():
                return False
        
        try:
            cursor = self.conn.cursor()
            
            # Проверяем, существует ли база данных
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (self.target_dbname,)
            )
            exists = cursor.fetchone()
            
            if not exists:
                # Создаем базу данных
                cursor.execute(
                    sql.SQL("CREATE DATABASE {}").format(
                        sql.Identifier(self.target_dbname)
                    )
                )
                logger.info(f"✅ База данных {self.target_dbname} создана")
            else:
                logger.info(f"✅ База данных {self.target_dbname} уже существует")
            
            cursor.close()
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка создания базы данных: {e}")
            return False
    
    def connect_to_dartserver(self):
        """Подключение к базе данных DartServer"""
        try:
            if self.conn:
                self.conn.close()
            
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                dbname=self.target_dbname,
                connect_timeout=5
            )
            logger.info(f"✅ Подключение к базе данных {self.target_dbname} установлено")
            return True
        except psycopg2.OperationalError as e:
            error_msg = str(e).strip()
            logger.error(f"❌ Ошибка подключения к базе данных {self.target_dbname}: {error_msg}")
            return False
        except Exception as e:
            error_msg = str(e).strip()
            logger.error(f"❌ Неожиданная ошибка подключения к базе данных {self.target_dbname}: {error_msg}")
            return False
    
    def create_tables(self):
        """Создание таблиц в базе данных"""
        if not self.conn:
            if not self.connect_to_dartserver():
                return False
        
        try:
            cursor = self.conn.cursor()
            
            # Создание таблицы Sport
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Sport (
                    Sport_ID SERIAL PRIMARY KEY,
                    Name VARCHAR(255) NOT NULL
                )
            """)
            
            # Создание таблицы Asser_types
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Asser_types (
                    Type_ID SERIAL PRIMARY KEY,
                    Name VARCHAR(255) NOT NULL
                )
            """)
            
            # Создание таблицы Pack
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Pack (
                    PackID SERIAL PRIMARY KEY,
                    Name VARCHAR(255) NOT NULL,
                    FK_Type_ID INTEGER REFERENCES Asser_types(Type_ID),
                    JsonFilePath VARCHAR(500),
                    FK_SportID INTEGER REFERENCES Sport(Sport_ID) NOT NULL,
                    UNIQUE(PackID, FK_SportID)
                )
            """)
            
            # Обновляем существующую таблицу, если JsonFilePath был INTEGER
            try:
                cursor.execute("""
                    ALTER TABLE Pack 
                    ALTER COLUMN JsonFilePath TYPE VARCHAR(500)
                    USING CASE 
                        WHEN JsonFilePath IS NULL THEN NULL
                        ELSE JsonFilePath::TEXT
                    END
                """)
                logger.info("✅ Столбец JsonFilePath обновлен на VARCHAR")
            except Exception:
                # Столбец уже VARCHAR или не существует - игнорируем
                pass
            
            self.conn.commit()
            cursor.close()
            logger.info("✅ Таблицы созданы успешно")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка создания таблиц: {e}")
            if self.conn:
                self.conn.rollback()
            return False
    
    def initialize_database(self):
        """Инициализация базы данных: создание БД и таблиц"""
        if not self.connect_to_postgres():
            return False
        
        if not self.create_database():
            return False
        
        if not self.connect_to_dartserver():
            return False
        
        if not self.create_tables():
            return False
        
        # Заполняем тестовыми данными
        if not self.seed_test_data():
            logger.warning("⚠️ Не удалось заполнить тестовыми данными")
        
        return True
    
    def seed_test_data(self):
        """Заполнение базы данных тестовыми данными при создании"""
        if not self.conn:
            return False
        
        try:
            cursor = self.conn.cursor()
            
            # Проверяем, есть ли уже данные
            cursor.execute("SELECT COUNT(*) FROM Sport")
            sport_count = cursor.fetchone()[0]
            
            # Если данные уже есть, очищаем таблицы для перезаполнения при создании
            if sport_count > 0:
                logger.info("🗑️ Очистка существующих данных для перезаполнения...")
                cursor.execute("DELETE FROM Pack")
                cursor.execute("DELETE FROM Asser_types")
                cursor.execute("DELETE FROM Sport")
                cursor.execute("ALTER SEQUENCE sport_sport_id_seq RESTART WITH 1")
                cursor.execute("ALTER SEQUENCE asser_types_type_id_seq RESTART WITH 1")
                cursor.execute("ALTER SEQUENCE pack_packid_seq RESTART WITH 1")
            
            # Вставляем тестовые данные в Sport
            cursor.execute("""
                INSERT INTO Sport (Name) VALUES
                ('Футбол'),
                ('Баскетбол'),
                ('Теннис'),
                ('Волейбол'),
                ('Хоккей'),
                ('Плавание'),
                ('Легкая атлетика'),
                ('Бокс')
            """)
            
            # Вставляем тестовые данные в Asser_types
            cursor.execute("""
                INSERT INTO Asser_types (Name) VALUES
                ('Оборудование'),
                ('Инвентарь'),
                ('Одежда'),
                ('Обувь'),
                ('Аксессуары'),
                ('Защитное снаряжение'),
                ('Тренажеры')
            """)
            
            # Получаем ID для создания связей
            cursor.execute("SELECT Sport_ID FROM Sport ORDER BY Sport_ID")
            sport_ids = [row[0] for row in cursor.fetchall()]
            
            cursor.execute("SELECT Type_ID FROM Asser_types ORDER BY Type_ID")
            type_ids = [row[0] for row in cursor.fetchall()]
            
            # Вставляем тестовые данные в Pack с большим разнообразием
            cursor.execute("""
                INSERT INTO Pack (Name, FK_Type_ID, JsonFilePath, FK_SportID) VALUES
                ('Футбольный мяч', %s, NULL, %s),
                ('Баскетбольный мяч', %s, NULL, %s),
                ('Теннисная ракетка', %s, NULL, %s),
                ('Волейбольная сетка', %s, NULL, %s),
                ('Хоккейная клюшка', %s, NULL, %s),
                ('Футбольная форма', %s, NULL, %s),
                ('Баскетбольные кроссовки', %s, NULL, %s),
                ('Теннисная сумка', %s, NULL, %s),
                ('Плавательные очки', %s, NULL, %s),
                ('Легкоатлетические шиповки', %s, NULL, %s),
                ('Боксерские перчатки', %s, NULL, %s),
                ('Футбольные щитки', %s, NULL, %s),
                ('Баскетбольная сетка', %s, NULL, %s),
                ('Хоккейная маска', %s, NULL, %s),
                ('Плавательная шапочка', %s, NULL, %s)
            """, (
                type_ids[1], sport_ids[0],  # Футбольный мяч
                type_ids[1], sport_ids[1],  # Баскетбольный мяч
                type_ids[0], sport_ids[2],  # Теннисная ракетка
                type_ids[0], sport_ids[3],  # Волейбольная сетка
                type_ids[0], sport_ids[4],  # Хоккейная клюшка
                type_ids[2], sport_ids[0],  # Футбольная форма
                type_ids[3], sport_ids[1],  # Баскетбольные кроссовки
                type_ids[4], sport_ids[2],  # Теннисная сумка
                type_ids[4], sport_ids[5],  # Плавательные очки
                type_ids[3], sport_ids[6],  # Легкоатлетические шиповки
                type_ids[5], sport_ids[7],  # Боксерские перчатки
                type_ids[5], sport_ids[0],  # Футбольные щитки
                type_ids[0], sport_ids[1],  # Баскетбольная сетка
                type_ids[5], sport_ids[4],  # Хоккейная маска
                type_ids[4], sport_ids[5]   # Плавательная шапочка
            ))
            
            self.conn.commit()
            cursor.close()
            logger.info("✅ Тестовые данные успешно добавлены (8 видов спорта, 7 типов активов, 15 пакетов)")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка заполнения тестовыми данными: {e}")
            if self.conn:
                self.conn.rollback()
            return False
    
    def get_all_sports(self) -> List[Dict[str, Any]]:
        """Получить все виды спорта"""
        if not self.conn:
            return []
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT Sport_ID, Name FROM Sport ORDER BY Sport_ID")
            rows = cursor.fetchall()
            cursor.close()
            return [{'Sport_ID': row[0], 'Name': row[1]} for row in rows]
        except Exception as e:
            logger.error(f"❌ Ошибка получения видов спорта: {e}")
            return []
    
    def get_all_asset_types(self) -> List[Dict[str, Any]]:
        """Получить все типы активов"""
        if not self.conn:
            return []
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT Type_ID, Name FROM Asser_types ORDER BY Type_ID")
            rows = cursor.fetchall()
            cursor.close()
            return [{'Type_ID': row[0], 'Name': row[1]} for row in rows]
        except Exception as e:
            logger.error(f"❌ Ошибка получения типов активов: {e}")
            return []
    
    def get_all_packs(self) -> List[Dict[str, Any]]:
        """Получить все пакеты с данными о связанных таблицах"""
        if not self.conn:
            return []
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT 
                    p.PackID,
                    p.Name,
                    p.FK_Type_ID,
                    at.Name AS TypeName,
                    p.JsonFilePath,
                    p.FK_SportID,
                    s.Name AS SportName
                FROM Pack p
                LEFT JOIN Asser_types at ON p.FK_Type_ID = at.Type_ID
                LEFT JOIN Sport s ON p.FK_SportID = s.Sport_ID
                ORDER BY p.PackID
            """)
            rows = cursor.fetchall()
            cursor.close()
            return [{
                'PackID': row[0],
                'Name': row[1],
                'FK_Type_ID': row[2],
                'TypeName': row[3],
                'JsonFilePath': row[4],
                'FK_SportID': row[5],
                'SportName': row[6]
            } for row in rows]
        except Exception as e:
            logger.error(f"❌ Ошибка получения пакетов: {e}")
            return []
    
    def add_sport(self, name: str) -> bool:
        """Добавить новый вид спорта"""
        if not self.conn:
            return False
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("INSERT INTO Sport (Name) VALUES (%s) RETURNING Sport_ID", (name,))
            self.conn.commit()
            sport_id = cursor.fetchone()[0]
            cursor.close()
            logger.info(f"✅ Вид спорта '{name}' добавлен с ID {sport_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка добавления вида спорта: {e}")
            if self.conn:
                self.conn.rollback()
            return False
    
    def add_asset_type(self, name: str) -> bool:
        """Добавить новый тип актива"""
        if not self.conn:
            return False
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("INSERT INTO Asser_types (Name) VALUES (%s) RETURNING Type_ID", (name,))
            self.conn.commit()
            type_id = cursor.fetchone()[0]
            cursor.close()
            logger.info(f"✅ Тип актива '{name}' добавлен с ID {type_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка добавления типа актива: {e}")
            if self.conn:
                self.conn.rollback()
            return False
    
    def add_pack(self, name: str, fk_type_id: int, json_file_path: str, fk_sport_id: int) -> bool:
        """Добавить новый пакет"""
        if not self.conn:
            return False
        
        try:
            cursor = self.conn.cursor()
            # Если fk_type_id равен 0 или None, вставляем NULL
            type_id_param = None if fk_type_id == 0 else fk_type_id
            json_path_param = json_file_path if json_file_path else None
            
            cursor.execute(
                "INSERT INTO Pack (Name, FK_Type_ID, JsonFilePath, FK_SportID) VALUES (%s, %s, %s, %s) RETURNING PackID",
                (name, type_id_param, json_path_param, fk_sport_id)
            )
            self.conn.commit()
            pack_id = cursor.fetchone()[0]
            cursor.close()
            logger.info(f"✅ Пакет '{name}' добавлен с ID {pack_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка добавления пакета: {e}")
            if self.conn:
                self.conn.rollback()
            return False
    
    def close(self):
        """Закрытие соединения с базой данных"""
        if self.conn:
            try:
                self.conn.close()
                logger.info("✅ Соединение с базой данных закрыто")
            except Exception as e:
                logger.error(f"❌ Ошибка закрытия соединения: {e}")
