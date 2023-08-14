SELECT
  'Average Answers per Question' AS metric,
  ROUND(AVG(answer_count), 2) AS average_answers_per_question
FROM (
  SELECT parent_id, COUNT(*) AS answer_count
  FROM `bigquery-public-data.stackoverflow.posts_answers`
  WHERE parent_id IN (
    SELECT id
    FROM `vital-program-390504.nath.final`
    WHERE creation_date BETWEEN '2018-07-01 00:00:00' AND '2022-06-30 23:59:59'
  )
  GROUP BY parent_id
);
