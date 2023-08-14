WITH TagCounts AS (
  SELECT id, ARRAY_LENGTH(SPLIT(tags, '|')) AS tag_count
  FROM `vital-program-390504.nath.final`
  WHERE creation_date BETWEEN '2018-07-01 00:00:00' AND '2022-06-30 23:59:59'
)

SELECT
  'Average Tags per Question' AS tag,
  ROUND(AVG(tag_count), 2) AS average_tags_per_question
FROM TagCounts;
