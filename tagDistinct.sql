WITH TagCounts AS (
  SELECT tag, COUNT(*) AS tag_count
  FROM (
    SELECT SPLIT(tags, '|') AS tags
    FROM `vital-program-390504.nath.final`
    WHERE  creation_date BETWEEN '2018-07-01 00:00:00' AND '2022-06-30 23:59:59'
  )
  CROSS JOIN UNNEST(tags) AS tag
  WHERE tag != 'github-actions'
  GROUP BY tag
)

SELECT
  'github-action' AS tag,
  COUNT(*) AS tag_count,
  (COUNT(*) * 100.0 / SUM(COUNT(*)) OVER ()) AS percentage
FROM `vital-program-390504.nath.final`
WHERE  creation_date BETWEEN '2018-07-01 00:00:00' AND '2022-06-30 23:59:59'
UNION ALL
SELECT
  tag,
  tag_count,
  (tag_count * 100.0 / SUM(tag_count) OVER ()) AS percentage
FROM TagCounts
ORDER BY tag_count DESC;
