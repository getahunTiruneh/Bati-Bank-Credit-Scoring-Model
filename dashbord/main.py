from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import io

# Load your trained model
logistic_model = joblib.load("logistic_regression.pkl")
# print(type(rfc_model)) 
app = FastAPI()

# Define the required columns for prediction
required_columns = [
    'CustomerId', 'Total_Transaction_Amount', 'Avg_Transaction_Amount',
    'Transaction_Count', 'Std_Transaction_Amount', 'Amount', 'Value',
    'PricingStrategy', 'ChannelId1'
]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the file contents and load it into a DataFrame
        contents = await file.read()
        test_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Check if all required columns exist in the uploaded CSV
        missing_columns = [col for col in required_columns if col not in test_df.columns]
        
        if missing_columns:
            return JSONResponse(status_code=400, content={"error": f"Missing columns: {', '.join(missing_columns)}"})

        # Ensure 'CustomerId' column is present
        if 'CustomerId' not in test_df.columns:
            test_df['CustomerId'] = range(1, len(test_df) + 1)

        # Make predictions (assumes 'CustomerId' is not part of the features used for prediction)
        features = test_df.drop(columns='CustomerId', errors='ignore')  # Drop CustomerId if it exists
        predictions = logistic_model.predict(features)

        # Create a results DataFrame with 'CustomerId' and predictions
        # results_df = pd.DataFrame({
        #     'CustomerId': test_df['CustomerId'],
        #     'Prediction': predictions
        # })

        # Return the results as JSON response
        return JSONResponse(content={"Predictions": predictions.tolist()})

    except pd.errors.ParserError:
        return JSONResponse(status_code=400, content={"error": "Failed to parse the CSV file."})
    except ValueError as ve:
        return JSONResponse(status_code=400, content={"error": f"Value error: {str(ve)}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"An error occurred: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)