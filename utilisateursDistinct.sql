SELECT DISTINCT
  u.id AS user_id,
  u.display_name AS user_name
FROM (
  SELECT id AS post_id, owner_user_id
  FROM `vital-program-390504.nath.final`
  WHERE creation_date BETWEEN '2018-07-01 00:00:00' AND '2022-06-30 23:59:59'
  
) AS q
LEFT JOIN (
  SELECT parent_id, owner_user_id
  FROM `bigquery-public-data.stackoverflow.posts_answers`
  WHERE creation_date BETWEEN '2018-07-01 00:00:00' AND '2022-06-30 23:59:59'
) AS a
ON q.post_id = a.parent_id
JOIN `bigquery-public-data.stackoverflow.users` AS u
ON (q.owner_user_id = u.id OR a.owner_user_id = u.id);
