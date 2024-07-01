CREATE TABLE IF NOT EXISTS ai_image_search.image_data (
    id UInt64,
    image_name String,
    image_embedding Array(Float32),
    faces Array(UInt64),
    PRIMARY KEY (id)
) ENGINE = MergeTree()
ORDER BY column1;

CREATE TABLE IF NOT EXISTS ai_image_search.face_data (
    id UInt64,
    face_name String,
    face_embedding Array(Float32),
    example_image_name String,
    PRIMARY KEY (id)
) ENGINE = MergeTree()