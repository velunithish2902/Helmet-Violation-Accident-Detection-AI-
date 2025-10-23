CREATE TABLE detections (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    camera_location VARCHAR(100),
    class_label VARCHAR(50),
    confidence REAL,
    bbox_coordinates FLOAT8[],
    s3_link TEXT
);

