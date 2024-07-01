CREATE DATABASE IF NOT EXISTS ai_image_search;

CREATE TABLE IF NOT EXISTS ai_image_search.image_data (
    id UInt64,
    image_name String,
    image_embedding Array(Float32),
    faces Array(UInt64),
    PRIMARY KEY (id)
) ENGINE = MergeTree();

CREATE TABLE IF NOT EXISTS ai_image_search.face_data (
    id UInt64,
    face_name String,
    face_embedding Array(Float32),
    example_image_name String,
    appeared_in_images Array(UInt64),
    PRIMARY KEY (id)
) ENGINE = MergeTree();