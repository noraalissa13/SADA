from json import encoder
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from tensorflow.keras.models import load_model  # For loading pre-trained LSTM model
from muselsl import stream
from pylsl import StreamInlet, resolve_byprop
from threading import Thread
import time
from bleak import BleakScanner
import asyncio
import logging
import nest_asyncio
import mne
from sklearn.preprocessing import StandardScaler

# Configure logging to display debug-level messages
logging.basicConfig(level=logging.DEBUG)

# Apply a workaround to allow nested asyncio loops (useful in Jupyter notebooks or similar environments)
nest_asyncio.apply()

# Create an instance of the FastAPI application
app = FastAPI()

# Global variables
is_streaming = False # Tracks if EEG streaming is currently active
recording_lock = asyncio.Lock() # Lock to prevent concurrent access to recording resources
muses = [] # List to store discovered Muse devices

# Define base classes for each database
RawBase = declarative_base()
PreprocessedBase = declarative_base()
PredictionsBase = declarative_base()

# Define database URLs
RAW_DATABASE_URL = "sqlite:///./raw_eeg_data.db"
PREPROCESSED_DATABASE_URL = "sqlite:///./preprocessed_eeg_data.db"
PREDICTIONS_DATABASE_URL = "sqlite:///./predictions.db"

# Create engines for each database
raw_engine = create_engine(RAW_DATABASE_URL)
preprocessed_engine = create_engine(PREPROCESSED_DATABASE_URL)
predictions_engine = create_engine(PREDICTIONS_DATABASE_URL)

# Create session makers for each database
RawSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=raw_engine)
PreprocessedSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=preprocessed_engine)
PredictionsSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=predictions_engine)

# Define models for each database
class RawEEGData(RawBase):
    __tablename__ = "raw_eeg_data"
    id = Column(Integer, primary_key=True, index=True) # Primary key
    Timestamp = Column(Integer, index=True) # Timestamp of the data
    TP9 = Column(Float) # EEG channel TP9
    AF7 = Column(Float) # EEG channel AF7
    AF8 = Column(Float) # EEG channel AF8
    TP10 = Column(Float) # EEG channel TP10
    RightAux = Column(Float, nullable=True)

class PreprocessedEEGData(PreprocessedBase):
    __tablename__ = "preprocessed_eeg_data"  
    id = Column(Integer, primary_key=True, index=True) # Primary key
    Timestamp = Column(Integer, index=True) # Timestamp of the data
    # Frequency bands for each channel
    delta0 = Column(Float)
    theta0 = Column(Float)
    alpha0 = Column(Float)
    beta0 = Column(Float)
    gamma0 = Column(Float)
    delta1 = Column(Float)
    theta1 = Column(Float)
    alpha1 = Column(Float)
    beta1 = Column(Float)
    gamma1 = Column(Float)
    delta2 = Column(Float)
    theta2 = Column(Float)
    alpha2 = Column(Float)
    beta2 = Column(Float)
    gamma2 = Column(Float)
    delta3 = Column(Float)
    theta3 = Column(Float)
    alpha3 = Column(Float)
    beta3 = Column(Float)
    gamma3 = Column(Float)
    

class PredictionData(PredictionsBase):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True) # Primary key
    timestamp = Column(Integer, index=True) # Timestamp of the prediction
    predicted_label = Column(String, index=True)  # Predicted label (e.g., 'low', 'moderate', 'high')
    attention_level = Column(Float) # Attention level prediction

# Create tables for each database
RawBase.metadata.create_all(bind=raw_engine)
PreprocessedBase.metadata.create_all(bind=preprocessed_engine)
PredictionsBase.metadata.create_all(bind=predictions_engine)

async def start_stream(address):
    """
    Starts asynchronous streaming from the Muse device using pylsl.
    """
    global is_streaming  # Use the global variable to track streaming status
    is_streaming = True
    try:
        # Run stream(address) in a separate thread
        def run_stream():
            stream(address)  # Start streaming from the Muse device

        stream_thread = Thread(target=run_stream)
        stream_thread.start()

        print(f"Connecting to Muse: {address}...")
        print("Looking for an EEG stream...")

        # Resolve the EEG stream from the Muse device
        streams = resolve_byprop("type", "EEG")

        if streams:
            print(f"Connected to EEG stream from {address}")

            # Start recording using the EEGRecorderToDB class
            recorder = EEGRecorderToDB()
            recorder.start_recording()

            # Keep the loop running while streaming
            while is_streaming:
                await asyncio.sleep(1)  # Prevent blocking other tasks
        else:
            print("No EEG stream found.")
    except Exception as e:
        print(f"Error during streaming: {e}")
    finally:
        is_streaming = False  # Reset the streaming flag when done
        print("Streaming stopped.")

# API Endpoints

@app.get("/attention-level/moving-average")
def fetch_moving_average(batch_size: int = 10):
    """
    Fetches the moving average of attention levels.
    """
    moving_average = get_moving_average(batch_size=batch_size)
    if moving_average is None:
        return {"message": "No attention level data available."}
    return {"moving_average": moving_average}

@app.websocket("/ws/eeg-stream")
async def eeg_stream(websocket: WebSocket):
    """
    WebSocket endpoint to handle EEG streaming communication with the client.
    """
    global is_streaming, muses  # Use global variables to track state
    await websocket.accept()

    try:
        while True:
            # Receive a message from the client (Flutter app)
            message = await websocket.receive_text()

            if message == "start_streaming":
                if is_streaming:
                    await websocket.send_text("Streaming is already in progress")
                    continue

                # Discover nearby Muse devices and start streaming from the first one found
                try:
                    devices = await BleakScanner.discover(timeout=10)
                    logging.debug(f"Devices discovered: {devices}")
                except Exception as e:
                    logging.error(f"Error discovering devices: {e}")
                    await websocket.send_text("Error discovering devices")
                    continue

                if devices is None or len(devices) == 0:
                    await websocket.send_text("No Muse devices found")
                    continue

                muses = [device for device in devices if device.name and "Muse" in device.name]

                if muses:
                    target_device = muses[0]  # Select the first Muse device
                    await websocket.send_text(f"Starting streaming from {target_device.name}")
                    await start_stream(target_device.address)  # Await the coroutine to start streaming
                    is_streaming = True
                else:
                    await websocket.send_text("No Muse devices found")

            elif message == "stop_streaming":
                # Logic to stop streaming (you can implement this part)
                await websocket.send_text("Stopping streaming...")

            else:
                await websocket.send_text("Invalid message")

            
                
    except WebSocketDisconnect:
        print("Client disconnected")
        is_streaming = False

class EEGRecorderToDB:
    def __init__(self):
        """
        Initialize the EEGRecorderToDB class with default attributes.
        """
        self.recording = False # Flag to indicate if recording is active
        self.inlet = None # Placeholder for the EEG data inlet stream
        self.thread = None # Thread for recording data
        self.db = None # Database session
        self.preprocessing_thread = None  # Thread for real-time data preprocessing
        self.prediction_thread = None  # Thread for real-time prediction

    def start_recording(self):
        """
        Start recording EEG data by launching the recording process in a separate thread.
        """
        self.recording = True # Set the recording flag to True
        self.thread = Thread(target=self.record_data_to_db) # Create a thread for data recording
        self.thread.start() # Start the thread

    def stop_recording(self):
        """
        Stop recording EEG data and clean up resources.
        """
        self.recording = False # Set the recording flag to False
        if self.thread:
            self.thread.join() # Wait for the recording thread to finish
        if self.db:
            self.db.close()  # Close the database session to release resources

    def record_data_to_db(self):
        """
        Continuously record EEG data from the Muse device and save it to the database.
        """
        try:
            print("Resolving EEG streams...")
            streams = resolve_byprop("type", "EEG") # Find EEG streams
            if not streams:
                print("No EEG streams found. Please make sure the Muse headband is connected.")
                return

            # Connect to the first available EEG stream
            self.inlet = StreamInlet(streams[0])
            print(f"Connected to EEG stream: {streams[0].name()}")

            # Open a database session for storing data
            self.db = RawSessionLocal()

            while self.recording:
                # Retrieve a chunk of EEG data along with timestamps
                chunk, timestamps = self.inlet.pull_chunk()
                if chunk and timestamps:
                    try:
                        # Flatten the chunk if it contains nested lists
                        eeg_values = [item for sublist in chunk for item in sublist] if isinstance(chunk[0], list) else chunk
                        
                        # Extract the first timestamp (or use the single value if not a list)
                        timestamp = timestamps[0] if isinstance(timestamps, list) else timestamps

                        # Combine the timestamp and EEG values into a single list
                        data_to_save = [timestamp] + eeg_values
                        
                        # Save the data to the database
                        save_eeg_data_to_db(self.db, data_to_save)

                        # Start the preprocessing thread if not already running
                        if not self.preprocessing_thread or not self.preprocessing_thread.is_alive():
                            self.preprocessing_thread = Thread(target=preprocess_data_in_real_time, daemon=True)
                            self.preprocessing_thread.start()

                        # Start the prediction thread if not already running
                        if not self.prediction_thread or not self.prediction_thread.is_alive():
                            self.prediction_thread = Thread(target=fetch_attention_level, daemon=True)
                            self.prediction_thread.start()

                    except Exception as e:
                        print(f"Error saving EEG data to database: {e}")
                        self.db.rollback()  # Rollback the database transaction in case of an error
                time.sleep(0.01)  # Add a short delay to control the sample rate

            print("Recording complete.")
        except Exception as e:
            print(f"Error during recording: {e}")
        finally:
            if self.db:
                self.db.close()  # Ensure the database session is closed

def save_eeg_data_to_db(db: Session, raw_data):
    """
    Save raw EEG data to the database.

    Args:
        db (Session): SQLAlchemy session for database operations.
        raw_data (list): Raw EEG data from the Muse device, including a timestamp and channel values.
    """
    try:
        # Extract timestamp and EEG channel values from the raw data
        timestamp = int(raw_data[0])  # The first element is the timestamp
        eeg_values = raw_data[1:]  # Remaining elements are EEG channel values

        # Create a new RawEEGData instance with the extracted values
        eeg_data = RawEEGData(
            Timestamp=timestamp,
            TP9=eeg_values[0], # TP9 channel value
            AF7=eeg_values[1],  # AF7 channel value
            AF8=eeg_values[2],  # AF8 channel value
            TP10=eeg_values[3],  # TP10 channel value
            RightAux=eeg_values[4] if len(eeg_values) > 4 else None,
        )

        # Add the data to the session and commit the transaction
        db.add(eeg_data)
        db.commit()
        db.refresh(eeg_data)  # Refresh the instance to get the updated data
    except Exception as e:
        logging.error(f"Error saving EEG data to database: {e}") # Log the error
    
def preprocess_data_in_real_time(batch_size=100, poll_interval=1):
    """
    Continuously preprocess raw EEG data in real time as it becomes available.

    Args:
        batch_size (int): The maximum number of records to process in a single batch.
        poll_interval (int): The time interval (in seconds) to wait before polling for new data.
    """
    try:
        # Initialize a new database session for saving preprocessed data.
        db = PreprocessedSessionLocal()

        # Keep track of the ID of the last processed raw EEG data record.
        last_processed_id = 0

        while True:
            # Open a separate session for querying raw EEG data.
            raw_db = RawSessionLocal()  

            # Fetch raw EEG data records that have not yet been processed.
            raw_data_records = (
                raw_db.query(RawEEGData)
                .filter(RawEEGData.id > last_processed_id)  # Filter for unprocessed records.
                .order_by(RawEEGData.id.asc()) # Process records in ascending order of ID.
                .limit(batch_size) # Limit the number of records fetched per batch.
                .all()
            )
        
            # Close the session for querying raw data to release resources.
            raw_db.close()

            if not raw_data_records:
                # If no new data is available, wait before polling again.
                print("No new data to process. Waiting for new data...")
                time.sleep(poll_interval)  # Wait before polling again
                continue

            # Prepare raw EEG data and timestamps for preprocessing.
            raw_data = []
            timestamps = []
            for record in raw_data_records:
                timestamps.append(record.Timestamp) # Collect timestamps from raw data.
                raw_data.append([record.TP9, record.AF7, record.AF8, record.TP10])  # Collect EEG values.

            # Convert raw data and timestamps to numpy arrays for processing.
            raw_data = np.array(raw_data).T # Transpose to match the shape (channels x time).
            timestamps = np.array(timestamps)

            # Create an MNE RawArray object for EEG data processing.
            channels = ['TP9', 'AF7', 'AF8', 'TP10'] # Define channel names.
            sfreq = 256  # Define the sampling frequency in Hz.
            info = mne.create_info(ch_names=channels, sfreq=sfreq, ch_types=['eeg'] * len(channels))
            raw = mne.io.RawArray(raw_data, info)

            # Apply a band-pass filter to isolate the frequency range of interest (1-30 Hz).
            raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=30.0, filter_length='auto')

            # Perform Independent Component Analysis (ICA) to remove artifacts from the data.
            ica = mne.preprocessing.ICA(n_components=len(channels), random_state=42, max_iter=200)
            ica.fit(raw_filtered) # Fit the ICA model to the filtered data.
            raw_cleaned = raw_filtered.copy()
            ica.apply(raw_cleaned) # Apply the ICA model to clean the data.

            # Extract features for predefined frequency bands (e.g., delta, theta, alpha, etc.).
            features = {'timestamp': timestamps}
            freq_bands = {
                'delta': (1, 4),
                'theta': (4, 8),
                'alpha': (8, 12),
                'beta': (12, 30),
                'gamma': (30, 45)
            }
            for channel_index in range(len(channels)):
                for band, freq_range in freq_bands.items():
                    # Filter data for the current frequency band and channel.
                    raw_band = raw_cleaned.copy().pick([channels[channel_index]]).filter(l_freq=freq_range[0], h_freq=freq_range[1], filter_length='auto')
                    band_data = raw_band.get_data()[0] # Extract the filtered data.
                    band_power = np.mean(band_data ** 2) # Calculate the power of the band.
                    band_power = np.log(band_power + 1e-6)  # Apply log transformation to stabilize values.
                    feature_name = f"{band}{channel_index}" # Generate feature name.
                    features[feature_name] = band_power

            # Convert the extracted features into a pandas DataFrame.
            df = pd.DataFrame(features)
            df = df.rename(columns={'timestamp': 'Timestamp'}) # Rename the timestamp column.

            # Save the preprocessed data to the database.
            for index, row in df.iterrows():
                # Extract timestamp and EEG feature values from the row.
                timestamp = int(row['Timestamp'])  # Assuming 'Timestamp' column is present
                eeg_values = [
                    row.get('delta0', None),
                    row.get('theta0', None),
                    row.get('alpha0', None),
                    row.get('beta0', None),
                    row.get('gamma0', None),
                    row.get('delta1', None),
                    row.get('theta1', None),
                    row.get('alpha1', None),
                    row.get('beta1', None),
                    row.get('gamma1', None),
                    row.get('delta2', None),
                    row.get('theta2', None),
                    row.get('alpha2', None),
                    row.get('beta2', None),
                    row.get('gamma2', None),
                    row.get('delta3', None),
                    row.get('theta3', None),
                    row.get('alpha3', None),
                    row.get('beta3', None),
                    row.get('gamma3', None)
                ]

                # Create a new database entry for preprocessed data.
                preprocessed_data = PreprocessedEEGData(
                    Timestamp=timestamp,
                    delta0=eeg_values[0],
                    theta0=eeg_values[1],
                    alpha0=eeg_values[2],
                    beta0=eeg_values[3],
                    gamma0=eeg_values[4],
                    delta1=eeg_values[5],
                    theta1=eeg_values[6],
                    alpha1=eeg_values[7],
                    beta1=eeg_values[8],
                    gamma1=eeg_values[9],
                    delta2=eeg_values[10],
                    theta2=eeg_values[11],
                    alpha2=eeg_values[12],
                    beta2=eeg_values[13],
                    gamma2=eeg_values[14],
                    delta3=eeg_values[15],
                    theta3=eeg_values[16],
                    alpha3=eeg_values[17],
                    beta3=eeg_values[18],
                    gamma3=eeg_values[19]
                )
                # Add the new entry to the database session.
                db.add(preprocessed_data)

            db.commit() # Commit the changes to save the data.
            
            # Update the ID of the last processed record.
            last_processed_id = raw_data_records[-1].id
            

    except Exception as e:
        # Handle any errors that occur during preprocessing.
        print(f"Error during real-time preprocessing: {e}")
        db.rollback() # Roll back the transaction in case of an error.
    finally:
        db.close()  # Ensure the database session is closed to free resources.

def start_preprocessing_thread():
    """
    Start the real-time preprocessing function in a new thread.
    """
    # Create a new thread to run the `preprocess_data_in_real_time` function.
    # The thread is set as a daemon so it will automatically close when the main program exits.
    preprocessing_thread = Thread(target=preprocess_data_in_real_time, daemon=True)

    # Start the thread to begin real-time preprocessing.
    preprocessing_thread.start()

    # Return the thread object for potential monitoring or control.
    return preprocessing_thread

def fetch_attention_level(batch_size=100):
    """
    Fetch processed EEG data, scale it, predict attention levels using the LSTM model, 
    and save predictions to the database with attention levels mapped based on confidence.
    """
    try:
            # Initialize the ID of the last processed record to ensure new data is fetched.
            last_processed_id = 0
            
            # Create a database session for saving predictions.
            db = PredictionsSessionLocal()

            # Load the pre-trained LSTM model for predicting attention levels.
            model = tf.keras.models.load_model("lstm_model.keras")
            # Initialize a standard scaler for data normalization (same as used during training).
            scaler = StandardScaler()

            # Define a mapping from predicted class labels to attention levels.
            label_to_attention = {
                "Low": 0.5, # Low attention level
                "Moderate": 1.5, # Moderate attention level
                "High": 3.0 # High attention level
            }

            while True:
                # Create a session for fetching preprocessed EEG data.
                processed_db = PreprocessedSessionLocal()

                # Query the database for preprocessed EEG data that hasn't been processed yet.
                processed_data_records = (
                    processed_db.query(PreprocessedEEGData)
                    .filter(PreprocessedEEGData.id > last_processed_id)  # Fetch records with IDs greater than the last processed ID.
                    .order_by(PreprocessedEEGData.id.asc())  # Sort records in ascending order of IDs.
                    .limit(batch_size) # Limit the number of records fetched in one batch.
                    .all()	
                )
                # Close the session after querying.
                processed_db.close()

                # Prepare the data for prediction.
                timestamps = []  # List to store timestamps of the records.
                features = [] # List to store feature vectors extracted from the records.

                # Extract features and timestamps from each record.
                for record in processed_data_records:
                    timestamps.append(record.Timestamp)
                    features.append([  # Append feature values for all channels and frequency bands.
                        record.delta0, record.theta0, record.alpha0, record.beta0, record.gamma0,
                        record.delta1, record.theta1, record.alpha1, record.beta1, record.gamma1,
                        record.delta2, record.theta2, record.alpha2, record.beta2, record.gamma2,
                        record.delta3, record.theta3, record.alpha3, record.beta3, record.gamma3
                    ])

                # Convert the extracted features into a DataFrame for easier handling.
                new_df = pd.DataFrame(features, columns=[
                    'delta0', 'theta0', 'alpha0', 'beta0', 'gamma0', 
                    'delta1', 'theta1', 'alpha1', 'beta1', 'gamma1', 
                    'delta2', 'theta2', 'alpha2', 'beta2', 'gamma2', 
                    'delta3', 'theta3', 'alpha3', 'beta3', 'gamma3'
                ])
                
                # Normalize the data using the scaler (same preprocessing as training).
                X_new = new_df[features]
                X_new = scaler.transform(X_new)  # Apply the same scaling as training data

                # Reshape the data for LSTM input (samples, time_steps, features).
                X_new_reshaped = X_new.reshape((X_new.shape[0], 1, X_new.shape[1]))

                # Predict attention levels using the LSTM model.
                predicted_labels = model.predict(X_new_reshaped)

                 # Decode the predicted labels into class names (e.g., "Low", "Moderate", "High").
                predicted_labels = encoder.inverse_transform(predicted_labels)
                predicted_labels = predicted_labels.ravel() # Flatten the array.

                # Create a DataFrame with the predictions for easier processing.
                results_df = pd.DataFrame(X_new, columns=features)
                results_df['Predicted_Label'] = predicted_labels  # Add the predicted labels to the DataFrame.

                # Save the predictions to the database in chunks.
                for index, row in results_df.iterrows():
                    # Extract the timestamp and predicted label for the current record.
                    timestamp = int(row['Timestamp'])  # Ensure the timestamp is an integer.
                    predicted_label = row['Predicted_Label']

                   # Map the predicted label to an attention level.
                    attention_level = label_to_attention.get(predicted_label, 0) # Default to 0 if label not found.

                    # Create a new PredictionData instance for the record.
                    prediction_data = PredictionData(
                        timestamp=timestamp,
                        predicted_label=predicted_label,
                        attention_level=attention_level
                    )
                    # Add the prediction to the database session.
                    db.add(prediction_data)

                # Commit the batch of predictions to the database.
                db.commit()

                # Update the last processed ID to the ID of the last record in the batch.
                last_processed_id = processed_data_records[-1].id

    except Exception as e:
            # Print an error message if an exception occurs.
            print(f"Error during prediction: {e}")
            # Rollback any uncommitted changes to the database.
            db.rollback()  # Rollback if an error occurs
    finally:
            db.close()  # Ensure the session is closed after use

def get_moving_average(batch_size=10):
    """
    Fetch the most recent attention levels and calculate the moving average.
    
    Args:
        batch_size (int): Number of recent predictions to consider for the moving average.
    
    Returns:
        float: The moving average of the attention levels, or None if no data is available.
    """
    # Initialize a session for querying the Predictions database.
    db = PredictionsSessionLocal()
    try:
        # Query the most recent `batch_size` attention levels from the database.
        predictions = (
            db.query(PredictionData.attention_level) # Select only the attention_level column.
            .order_by(PredictionData.timestamp.desc()) # Sort by timestamp in descending order (most recent first).
            .limit(batch_size) # Limit the query to the specified batch size.
            .all() # Fetch all results from the query.
        )
        
        # Extract attention levels from the query result, ensuring no `None` values are included.
        attention_levels = [p.attention_level for p in predictions if p.attention_level is not None]

        # Check if there are any valid attention levels.
        if not attention_levels:
            return None  # Return None if no data is available.

        # Generate linearly decreasing weights for the moving average calculation.
        # For example, if there are 5 attention levels, weights will be [1.0, 0.75, 0.5, 0.25, 0.0].
        weights = np.linspace(1, 0, len(attention_levels))  # Linearly decreasing weights

        # Compute the weighted moving average using the attention levels and the weights.
        moving_average = np.average(attention_levels, weights=weights)

        # Return the calculated moving average.
        return moving_average
    finally:
        # Ensure the database session is properly closed, even if an exception occurs.
        db.close()