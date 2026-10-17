CREATE TABLE IF NOT EXISTS poem_vectors (
  poem_id INTEGER PRIMARY KEY REFERENCES poems(id) ON DELETE CASCADE,
  embedding vector(1024) NOT NULL
);
CREATE INDEX IF NOT EXISTS poem_embedding_stage_lookup
  ON poem_embedding_stage (author, title, doc_md5);

CREATE OR REPLACE PROCEDURE fill_poem_vectors(batch int DEFAULT 2000)
LANGUAGE plpgsql AS $$
DECLARE
  lo int;
  hi int;
  start_id int;
  end_id int;
  n int;
  total int := 0;
BEGIN
  PERFORM set_config('statement_timeout', '0', false);
  PERFORM set_config('synchronous_commit', 'off', false);
  PERFORM set_config('jit', 'off', false);
  PERFORM set_config('enable_hashjoin', 'off', false);
  PERFORM set_config('enable_mergejoin', 'off', false);
  SELECT MIN(id), MAX(id) INTO lo, hi FROM poems;
  start_id := lo;
  WHILE start_id <= hi LOOP
    end_id := LEAST(start_id + batch - 1, hi);
    INSERT INTO poem_vectors (poem_id, embedding)
    SELECT p.id, s.embedding
    FROM poems p
    JOIN poem_embedding_stage s
      ON p.author = s.author
     AND p.title = s.title
     AND md5(p.document) = s.doc_md5
    WHERE p.id BETWEEN start_id AND end_id
    ON CONFLICT (poem_id) DO NOTHING;
    GET DIAGNOSTICS n = ROW_COUNT;
    total := total + n;
    COMMIT;
    RAISE NOTICE 'vectors % - % batch=% total=%', start_id, end_id, n, total;
    start_id := start_id + batch;
  END LOOP;
  RAISE NOTICE 'fill done total=%', total;
END;
$$;
