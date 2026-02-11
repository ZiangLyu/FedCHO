const express = require('express');
const mysql = require('mysql2');
const cors = require('cors');
const bodyParser = require('body-parser');

const app = express();
const port = 3000;

app.use(bodyParser.json({ limit: '10000mb' }));
app.use(bodyParser.urlencoded({ limit: '10000mb', extended: true }));
app.use(cors());

const dbName = `terminal_${Date.now()}`;

const baseDb = mysql.createConnection({
    host: 'localhost',
    user: 'root',
    password: '998699'
});

// 初始化数据库：创建 Visit + Terminal 表
async function initDatabase() {
    try {
        await new Promise((resolve, reject) => {
            baseDb.connect(err => err ? reject(`基础连接失败: ${err.message}`) : resolve());
        });

        await new Promise((resolve, reject) => {
            baseDb.query(`CREATE DATABASE IF NOT EXISTS \`${dbName}\``, err =>
                err ? reject(`创建数据库失败: ${err.message}`) : resolve()
            );
        });

        baseDb.end();

        const db = mysql.createConnection({
            host: 'localhost',
            user: 'root',
            password: '998699',
            database: dbName
        });

        await new Promise((resolve, reject) => {
            db.connect(err => err ? reject(`连接新数据库失败: ${err.message}`) : resolve());
        });

        // 创建 Visit 表
        const createVisitTable = `
            CREATE TABLE IF NOT EXISTS Visit (
                拜访记录编号 VARCHAR(50),
                拜访开始时间 VARCHAR(50),
                拜访结束时间 VARCHAR(50),
                拜访人 VARCHAR(50),
                客户名称 VARCHAR(100),
                客户编码 VARCHAR(50),
                拜访用时 INT,
                INDEX idx_visit_customer (客户编码),
                INDEX idx_visit_time (拜访用时),
                INDEX idx_visit_start (拜访开始时间)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        `;

        // 创建 Terminal 表（客户编码唯一，避免 JOIN 膨胀）
        const createTerminalTable = `
            CREATE TABLE IF NOT EXISTS Terminal (
                客户编码 VARCHAR(50),
                所属片区 VARCHAR(100),
                所属大区 VARCHAR(100),
                UNIQUE INDEX idx_terminal_customer (客户编码)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        `;

        await new Promise((resolve, reject) => {
            db.query(createVisitTable, err => err ? reject(`创建 Visit 表失败: ${err.message}`) : resolve());
        });

        await new Promise((resolve, reject) => {
            db.query(createTerminalTable, err => err ? reject(`创建 Terminal 表失败: ${err.message}`) : resolve());
        });

        app.set('db', db);
        console.log(`数据库初始化完成: ${dbName}`);
        console.log('Visit 和 Terminal 表已创建');

    } catch (error) {
        console.error('数据库初始化失败:', error);
        process.exit(1);
    }
}

// 上传 Visit
app.post('/uploadVisit', (req, res) => {
    const db = app.get('db');
    const records = req.body.records;

    if (!Array.isArray(records) || records.length === 0) {
        return res.status(400).json({ success: false, error: '无效的 Visit 数据' });
    }

    const values = records.map(r => [
        r.拜访记录编号 || null,
        r.拜访开始时间 || null,
        r.拜访结束时间 || null,
        r.拜访人 || null,
        r.客户名称 || null,
        r.客户编码 || null,
        typeof r.拜访用时 === 'string' ? parseInt(r.拜访用时) || 0 : (r.拜访用时 || 0)
    ]);

    const sql = 'INSERT INTO Visit (拜访记录编号, 拜访开始时间, 拜访结束时间, 拜访人, 客户名称, 客户编码, 拜访用时) VALUES ?';
    db.query(sql, [values], (err, result) => {
        if (err) {
            console.error('Visit 插入失败:', err);
            res.status(500).json({ success: false, error: err.message });
        } else {
            res.json({ success: true, message: `成功导入 ${result.affectedRows} 条 Visit 记录` });
        }
    });
});

// 上传 Terminal（使用 INSERT IGNORE 避免重复客户编码报错）
app.post('/uploadTerminal', (req, res) => {
    const db = app.get('db');
    const records = req.body.records;

    if (!Array.isArray(records) || records.length === 0) {
        return res.status(400).json({ success: false, error: '无效的 Terminal 数据' });
    }

    const values = records.map(r => [
        r.客户编码 || null,
        r.所属片区 || null,
        r.所属大区 || null
    ]);

    const sql = 'INSERT IGNORE INTO Terminal (客户编码, 所属片区, 所属大区) VALUES ?';
    db.query(sql, [values], (err, result) => {
        if (err) {
            console.error('Terminal 插入失败:', err);
            res.status(500).json({ success: false, error: err.message });
        } else {
            res.json({ success: true, message: `成功导入 ${result.affectedRows} 条 Terminal 记录（已自动去重）` });
        }
    });
});

// 查询合并后的数据（LEFT JOIN Visit + Terminal）-- 使用 DISTINCT 去重
app.get('/getMinutes', (req, res) => {
    const db = app.get('db');
    let {
        maxMinutes = 5,
        visitor = '',
        customerName = '',
        customerCode = '',
        startDate = '',
        endDate = '',
        area = '',
        region = ''
    } = req.query;

    maxMinutes = parseInt(maxMinutes) || 5;

    // 构建 LEFT JOIN 查询条件
    let conditions = ['v.`拜访用时` <= ?'];
    let params = [maxMinutes];

    if (visitor) {
        conditions.push('v.`拜访人` LIKE ?');
        params.push(`%${visitor}%`);
    }
    if (customerName) {
        conditions.push('v.`客户名称` LIKE ?');
        params.push(`%${customerName}%`);
    }
    if (customerCode) {
        conditions.push('v.`客户编码` LIKE ?');
        params.push(`%${customerCode}%`);
    }
    if (startDate) {
        conditions.push('v.`拜访开始时间` >= ?');
        params.push(startDate);
    }
    if (endDate) {
        conditions.push('v.`拜访开始时间` <= ?');
        params.push(endDate + ' 23:59:59');
    }
    // 按所属片区筛选
    if (area) {
        conditions.push('t.`所属片区` LIKE ?');
        params.push(`%${area}%`);
    }
    // 按所属大区筛选
    if (region) {
        conditions.push('t.`所属大区` LIKE ?');
        params.push(`%${region}%`);
    }

    const whereClause = conditions.join(' AND ');

    // 使用 DISTINCT 去除完全相同的重复行
    const sql = `
        SELECT DISTINCT
            v.拜访记录编号,
            v.拜访开始时间,
            v.拜访结束时间,
            v.拜访人,
            v.客户名称,
            v.客户编码,
            v.拜访用时,
            t.所属片区,
            t.所属大区
        FROM Visit v
        LEFT JOIN Terminal t ON v.客户编码 = t.客户编码
        WHERE ${whereClause}
        ORDER BY v.\`拜访用时\` ASC, v.\`拜访开始时间\` DESC;
    `;

    db.query(sql, params, (err, results) => {
        if (err) {
            console.error('查询合并数据失败:', err);
            res.status(500).json({ success: false, error: err.message });
        } else {
            res.json({ success: true, data: results });
        }
    });
});

// 获取所有片区列表（用于前端下拉选择）
app.get('/getAreas', (req, res) => {
    const db = app.get('db');
    const sql = 'SELECT DISTINCT 所属片区 FROM Terminal WHERE 所属片区 IS NOT NULL AND 所属片区 != "" ORDER BY 所属片区';
    db.query(sql, (err, results) => {
        if (err) {
            res.status(500).json({ success: false, error: err.message });
        } else {
            res.json({ success: true, data: results.map(r => r.所属片区) });
        }
    });
});

// 获取所有大区列表（用于前端下拉选择）
app.get('/getRegions', (req, res) => {
    const db = app.get('db');
    const sql = 'SELECT DISTINCT 所属大区 FROM Terminal WHERE 所属大区 IS NOT NULL AND 所属大区 != "" ORDER BY 所属大区';
    db.query(sql, (err, results) => {
        if (err) {
            res.status(500).json({ success: false, error: err.message });
        } else {
            res.json({ success: true, data: results.map(r => r.所属大区) });
        }
    });
});

// ============ 清理：删除整个数据库 ============
app.post('/cleanup', async (req, res) => {
    try {
        await dropDatabase();
        res.json({ success: true, message: `数据库 ${dbName} 及其所有表已删除` });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// 删除数据库的核心函数
async function dropDatabase() {
    const db = app.get('db');
    if (db) {
        try { db.end(); } catch (e) { /* 忽略 */ }
    }

    const cleanupDb = mysql.createConnection({
        host: 'localhost',
        user: 'root',
        password: '998699'
    });

    await new Promise((resolve, reject) => {
        cleanupDb.connect(err => err ? reject(err) : resolve());
    });

    await new Promise((resolve, reject) => {
        cleanupDb.query(`DROP DATABASE IF EXISTS \`${dbName}\``, err => {
            if (err) reject(err);
            else resolve();
        });
    });

    cleanupDb.end();
    console.log(`数据库已删除: ${dbName}`);
}

// 进程退出时自动清理数据库
function setupProcessCleanup() {
    let cleaning = false;

    async function handleExit(signal) {
        if (cleaning) return;
        cleaning = true;
        console.log(`\n收到 ${signal} 信号，正在清理数据库...`);
        try {
            await dropDatabase();
            console.log('清理完成，进程退出');
        } catch (error) {
            console.error('清理时出错:', error.message);
        }
        process.exit(0);
    }

    process.on('SIGINT', () => handleExit('SIGINT'));
    process.on('SIGTERM', () => handleExit('SIGTERM'));
}

initDatabase().then(() => {
    setupProcessCleanup();
    app.listen(port, () => {
        console.log(`服务器运行在 http://localhost:${port}`);
        console.log(`当前数据库: ${dbName}`);
    });
});
