CREATE DATABASE IF NOT EXISTS ai_image_search;

CREATE TABLE IF NOT EXISTS ai_image_search.image_data (
    id UInt64,
    image_name String,
    image_embedding Array(Float32),
) ENGINE = MergeTree
PRIMARY KEY (id);

CREATE TABLE IF NOT EXISTS ai_image_search.face_data (
    id UInt64,
    face_name String,
    face_embedding Array(Float32),
    example_image_name String,
    
) ENGINE = ReplacingMergeTree
PRIMARY KEY (id);

CREATE TABLE IF NOT EXISTS ai_image_search.image_faces (
    image_id UInt64,
    face_id UInt64,
    
) ENGINE = MergeTree
PRIMARY KEY (image_id, face_id)
