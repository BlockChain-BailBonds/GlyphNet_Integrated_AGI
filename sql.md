# Codex — sql

### ι

```sql
INSERT INTO {0}({1}) VALUES({2});
```

### μ

```sql
UPDATE {0} SET {1}{2};
```

### σ

```sql
SELECT {0} FROM {1}{2};
```

### τ

```sql
DELETE FROM {0}{1};
```

### ω

```sql
 WHERE {0}
```

### Ωsql_alter_add

```sql
ALTER TABLE {0} ADD COLUMN {1} {2};
```

### Ωsql_create_table

```sql
CREATE TABLE {0} ({1});
```

### Ωsql_group_having

```sql
SELECT {0}, {1} FROM {2} GROUP BY {0} HAVING {3};
```

### Ωsql_join_on

```sql
SELECT {0} FROM {1} JOIN {2} ON {3};
```

### Ωsql_users_orders_demo

```sql
CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, total REAL);
INSERT INTO users(id,name) VALUES (1,'Ada'),(2,'Linus');
INSERT INTO orders(id,user_id,total) VALUES (1,1,12.5),(2,1,7.0),(3,2,20.0);
SELECT u.name, SUM(o.total) AS spend
FROM users u JOIN orders o ON u.id=o.user_id
GROUP BY u.name
ORDER BY spend DESC;

```
