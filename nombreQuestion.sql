SELECT count(*)
FROM `bigquery-public-data.stackoverflow.posts_questions`
WHERE 
    creation_date BETWEEN '2018-07-01 00:00:00' AND '2022-06-30 23:59:59'
    AND (
        LOWER(title) LIKE '%github actions%' 
        OR LOWER(body) LIKE '%github actions%' 
        
        OR id IN (
            SELECT id
            FROM (
                SELECT id, SPLIT(tags, '|') tags
                FROM `bigquery-public-data.stackoverflow.posts_questions`
                WHERE creation_date BETWEEN '2018-07-01 00:00:00' AND '2022-06-30 23:59:59'
            ) 
            CROSS JOIN UNNEST(tags) flattened_tags
            WHERE LOWER(flattened_tags) = 'github-actions'
        )
    )

