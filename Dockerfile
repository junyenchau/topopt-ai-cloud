FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files (make sure final_topology_generator.h5 is in the models/ folder!)
COPY . .

# Use Cloud Run's default port variable
ENV PORT=8080
EXPOSE $PORT

CMD streamlit run src/app.py --server.port=$PORT --server.address=0.0.0.0