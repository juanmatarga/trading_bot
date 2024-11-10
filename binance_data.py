from binance.client import Client
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='trading_bot.log'
)

class BinanceDataLoader:
    def __init__(self):
        """
        Inicializa el cargador de datos de Binance.
        Timeframe ajustado a 4 horas para mejor granularidad.
        """
        try:
            self.symbol = 'BTCUSDT'
            self.timeframe = Client.KLINE_INTERVAL_4HOUR  # Cambiado a 4 horas
            self.period = '1 year'
            self.limit = 2190  # Aumentado para tener más datos históricos
            
            # Crear cliente de Binance (solo datos públicos)
            self.client = Client(None, None)
            
            # Crear directorio de datos
            os.makedirs('data', exist_ok=True)
            
            logging.info(f"BinanceDataLoader inicializado para {self.symbol} "
                        f"con timeframe de 4 horas")
            
        except Exception as e:
            logging.error(f"Error en inicialización de BinanceDataLoader: {str(e)}")
            raise

    def _cargar_datos_locales(self):
        """
        Carga datos desde archivo local si existe.
        
        Returns:
            pandas.DataFrame: DataFrame con datos históricos o None si no existe
        """
        try:
            filename = f'data/{self.symbol}_{self.timeframe}.csv'
            
            if not os.path.exists(filename):
                logging.info("No se encontró archivo de datos local")
                return None
                
            # Cargar datos
            df = pd.read_csv(filename)
            
            # Convertir timestamp a datetime y establecer como índice
            df['timestamp'] = pd.to_datetime(df.index)
            df.set_index('timestamp', inplace=True)
            
            # Verificar si los datos están desactualizados
            if self._datos_desactualizados(df):
                logging.info("Datos locales desactualizados")
                return None
                
            logging.info(f"Datos locales cargados: {len(df)} registros")
            return df
            
        except Exception as e:
            logging.error(f"Error al cargar datos locales: {str(e)}")
            return None
            
    def obtener_datos_historicos(self):
        """
        Obtiene datos históricos de Binance con validaciones mejoradas.
        
        Returns:
            pandas.DataFrame: DataFrame con datos históricos
        """
        try:
            # Intentar cargar datos locales primero
            df_local = self._cargar_datos_locales()
            
            if df_local is not None and not self._datos_desactualizados(df_local):
                logging.info(f"Usando datos locales existentes: {len(df_local)} registros")
                return df_local
                
            # Si no hay datos locales o están desactualizados, descargar nuevos
            logging.info("Descargando nuevos datos de Binance")
            
            # Calcular fecha inicial (1 año atrás)
            fecha_inicial = datetime.now() - timedelta(days=365)
            
            # Obtener klines en lotes para evitar límites
            klines = []
            fecha_actual = fecha_inicial
            
            while fecha_actual < datetime.now():
                lote = self.client.get_historical_klines(
                    symbol=self.symbol,
                    interval=self.timeframe,
                    start_str=fecha_actual.strftime("%d %b %Y %H:%M:%S"),
                    end_str=(fecha_actual + timedelta(days=30)).strftime("%d %b %Y %H:%M:%S"),
                    limit=1000
                )
                
                if lote:
                    klines.extend(lote)
                    fecha_actual += timedelta(days=30)
                    logging.info(f"Obtenidos {len(lote)} registros para {fecha_actual}")
                else:
                    break
            
            if not klines:
                raise ValueError("No se obtuvieron datos de Binance")
                
            # Convertir a DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                'ignore'
            ])
            
            # Procesar datos
            df = self._procesar_datos(df)
            
            # Validar datos procesados
            self._validar_datos(df)
            
            # Guardar datos localmente
            self._guardar_datos_locales(df)
            
            logging.info(f"Datos descargados y procesados: {len(df)} registros")
            return df
            
        except Exception as e:
            logging.error(f"Error al obtener datos históricos: {str(e)}")
            return pd.DataFrame()
            
    def _datos_desactualizados(self, df):
        """
        Verifica si los datos locales están desactualizados.
        Ajustado para considerar el intervalo de 4 horas.
        
        Args:
            df (pandas.DataFrame): DataFrame con datos locales
            
        Returns:
            bool: True si los datos están desactualizados
        """
        try:
            if df.empty:
                return True
                
            ultima_fecha = df.index[-1]
            tiempo_actual = pd.Timestamp.now()
            
            # Considerar desactualizado si han pasado más de 4 horas
            return (tiempo_actual - ultima_fecha) > timedelta(hours=4)
            
        except Exception as e:
            logging.error(f"Error al verificar actualización de datos: {str(e)}")
            return True
            
    def _procesar_datos(self, df):
        """
        Procesa los datos crudos de Binance.
        
        Args:
            df (pandas.DataFrame): DataFrame con datos crudos
            
        Returns:
            pandas.DataFrame: DataFrame procesado
        """
        try:
            # Convertir timestamp a datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Convertir columnas numéricas
            columnas_numericas = ['open', 'high', 'low', 'close', 'volume']
            for col in columnas_numericas:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Establecer índice
            df.set_index('timestamp', inplace=True)
            
            # Seleccionar solo columnas necesarias
            df = df[columnas_numericas]
            
            # Eliminar duplicados y ordenar
            df = df.drop_duplicates()
            df.sort_index(inplace=True)
            
            # Eliminar filas con valores nulos
            df.dropna(inplace=True)
            
            # Verificar intervalos consistentes de 4 horas
            self._verificar_intervalos(df)
            
            return df
            
        except Exception as e:
            logging.error(f"Error al procesar datos: {str(e)}")
            raise
            
    def _verificar_intervalos(self, df):
        """
        Verifica que los intervalos sean consistentes de 4 horas.
        
        Args:
            df (pandas.DataFrame): DataFrame a verificar
        """
        try:
            # Calcular diferencias entre timestamps
            diff_tiempo = df.index.to_series().diff()
            
            # Verificar que la mayoría de los intervalos sean de 4 horas
            intervalo_esperado = pd.Timedelta(hours=4)
            intervalos_incorrectos = diff_tiempo[
                diff_tiempo != intervalo_esperado
            ]
            
            if len(intervalos_incorrectos) > len(df) * 0.05:  # Más del 5% incorrectos
                logging.warning(
                    f"Detectados {len(intervalos_incorrectos)} intervalos "
                    f"inconsistentes de 4 horas"
                )
                
        except Exception as e:
            logging.error(f"Error al verificar intervalos: {str(e)}")
            
    def _validar_datos(self, df):
        """
        Valida la integridad de los datos.
        
        Args:
            df (pandas.DataFrame): DataFrame a validar
        """
        try:
            # Verificar valores nulos
            if df.isnull().any().any():
                raise ValueError("Los datos contienen valores nulos")
            
            # Asegurarse de que todas las columnas numéricas sean float
            columnas_numericas = ['open', 'high', 'low', 'close', 'volume']
            for col in columnas_numericas:
                df[col] = df[col].astype(float)
                
            # Verificar valores negativos en columnas de precio
            for col in ['open', 'high', 'low', 'close']:
                if (df[col].astype(float) < 0).any():
                    raise ValueError(f"La columna {col} contiene valores negativos")
            
            # Verificar orden de precios (high debe ser mayor que low)
            invalid_prices = df[df['high'].astype(float) < df['low'].astype(float)].index
            if not invalid_prices.empty:
                raise ValueError(f"Precios inválidos encontrados en: {invalid_prices}")
            
            # Verificar volumen
            if (df['volume'].astype(float) < 0).any():
                raise ValueError("Volúmenes negativos encontrados")
                
            if (df['volume'].astype(float) == 0).any():
                logging.warning("Detectados períodos con volumen cero")
            
            logging.info("Validación de datos completada exitosamente")
            
        except Exception as e:
            logging.error(f"Error en validación de datos: {str(e)}")
            raise
            
    def _guardar_datos_locales(self, df):
        """
        Guarda los datos en archivo local con metadata.
        
        Args:
            df (pandas.DataFrame): DataFrame a guardar
        """
        try:
            filename = f'data/{self.symbol}_{self.timeframe}.csv'
            df.to_csv(filename)
            
            # Guardar metadata
            metadata = {
                'last_update': datetime.now().isoformat(),
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'rows': len(df),
                'fecha_inicial': df.index[0].isoformat(),
                'fecha_final': df.index[-1].isoformat()
            }
            
            with open(f'data/{self.symbol}_{self.timeframe}_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=4)
                
            logging.info(f"Datos guardados localmente en {filename}")
            
        except Exception as e:
            logging.error(f"Error al guardar datos locales: {str(e)}")
            raise

def main():
    """Función principal para pruebas"""
    try:
        loader = BinanceDataLoader()
        df = loader.obtener_datos_historicos()
        
        if df.empty:
            logging.error("No se pudieron obtener datos")
            return
            
        print(f"Datos obtenidos exitosamente: {len(df)} registros")
        print("\nPrimeras 5 filas:")
        print(df.head())
        print("\nÚltimas 5 filas:")
        print(df.tail())
        print(f"\nIntervalo promedio: {df.index.to_series().diff().mean()}")
        
    except Exception as e:
        logging.error(f"Error en la ejecución principal: {str(e)}")
        raise

if __name__ == "__main__":
    main()