WITH NEWTABLE AS (
    SELECT S.NAME AS NAME_ST, G.GRADE AS VGRADE, S.MARKS AS NEW_MARK
    FROM STUDENTS AS S
    INNER JOIN GRADES AS G
    ON S.MARKS BETWEEN G.MIN_MARK AND  G.MAX_MARK
    ORDER BY VGRADE DESC, NAME_ST ASC, NEW_MARK ASC
)

SELECT 
    CASE
        WHEN VGRADE > 7 THEN NAME_ST
        ELSE NULL
    END
    , VGRADE, NEW_MARK
FROM NEWTABLE
