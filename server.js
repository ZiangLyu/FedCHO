const express = require('express');
const mysql = require('mysql2');
const cors = require('cors');
const bodyParser = require('body-parser');

const app = express();
const port = 3000;

app.use(bodyParser.json({ limit: '10000mb' }));
app.use(bodyParser.urlencoded({ limit: '10000mb', extended: true }));
app.use(cors());

let dbName = `terminal_${Date.now()}`;

// ============ Database state management ============
// clientConnected: becomes true after the first heartbeat is received
// cleanedUp: becomes true after the database has been dropped
// When cleanedUp is true and a new request arrives, the database is automatically re-created
let lastHeartbeat = 0;
let clientConnected = false;
let cleanedUp = false;
const HEARTBEAT_TIMEOUT = 90000;
const HEARTBEAT_CHECK_INTERVAL = 15000;

const DB_CONFIG = {
    host: 'localhost',
    user: 'root',
    password: '998699'
};

// ============ Database initialization ============
async function initDatabase() {
    try {
        // Generate a new unique database name each time
        dbName = `terminal_${Date.now()}`;

        const baseDb = mysql.createConnection(DB_CONFIG);

        await new Promise((resolve, reject) => {
            baseDb.connect(err => err ? reject(`Base connection failed: ${err.message}`) : resolve());
        });

        await new Promise((resolve, reject) => {
            baseDb.query(`CREATE DATABASE IF NOT EXISTS \`${dbName}\``, err =>
                err ? reject(`Failed to create database: ${err.message}`) : resolve()
            );
        });

        baseDb.end();

        const db = mysql.createConnection({ ...DB_CONFIG, database: dbName });

        await new Promise((resolve, reject) => {
            db.connect(err => err ? reject(`Failed to connect to new database: ${err.message}`) : resolve());
        });

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

        const createTerminalTable = `
            CREATE TABLE IF NOT EXISTS Terminal (
                客户编码 VARCHAR(50),
                所属片区 VARCHAR(100),
                所属大区 VARCHAR(100),
                UNIQUE INDEX idx_terminal_customer (客户编码)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        `;

        await new Promise((resolve, reject) => {
            db.query(createVisitTable, err => err ? reject(`Failed to create Visit table: ${err.message}`) : resolve());
        });

        await new Promise((resolve, reject) => {
            db.query(createTerminalTable, err => err ? reject(`Failed to create Terminal table: ${err.message}`) : resolve());
        });

        app.set('db', db);
        cleanedUp = false;
        console.log(`Database initialized: ${dbName}`);
        console.log('Visit and Terminal tables created');

    } catch (error) {
        console.error('Database initialization failed:', error);
        throw error;
    }
}

// Ensure database exists before handling data requests.
// If it was previously cleaned up, re-create it automatically.
async function ensureDatabase() {
    if (!cleanedUp) return true;
    console.log('Database was cleaned up. Re-creating for new session...');
    try {
        await initDatabase();
        return true;
    } catch (error) {
        console.error('Failed to re-create database:', error);
        return false;
    }
}

// ============ Heartbeat endpoint ============
app.get('/api/audit_visit/search_time/heartbeat', (req, res) => {
    lastHeartbeat = Date.now();
    if (!clientConnected) {
        clientConnected = true;
        console.log('Client connected, heartbeat tracking started');
    }
    res.json({ success: true, timestamp: lastHeartbeat });
});

// Upload Visit records
app.post('/api/audit_visit/search_time/uploadVisit', async (req, res) => {
    lastHeartbeat = Date.now();
    clientConnected = true;

    if (!(await ensureDatabase())) {
        return res.status(500).json({ success: false, error: 'Database unavailable, please refresh the page and try again' });
    }

    const db = app.get('db');
    const records = req.body.records;

    if (!Array.isArray(records) || records.length === 0) {
        return res.status(400).json({ success: false, error: 'Invalid Visit data provided' });
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
            console.error('Failed to insert Visit records:', err);
            res.status(500).json({ success: false, error: err.message });
        } else {
            res.json({ success: true, message: `${result.affectedRows} Visit records imported successfully` });
        }
    });
});

// Upload Terminal records
app.post('/api/audit_visit/search_time/uploadTerminal', async (req, res) => {
    lastHeartbeat = Date.now();
    clientConnected = true;

    if (!(await ensureDatabase())) {
        return res.status(500).json({ success: false, error: 'Database unavailable, please refresh the page and try again' });
    }

    const db = app.get('db');
    const records = req.body.records;

    if (!Array.isArray(records) || records.length === 0) {
        return res.status(400).json({ success: false, error: 'Invalid Terminal data provided' });
    }

    const values = records.map(r => [
        r.客户编码 || null,
        r.所属片区 || null,
        r.所属大区 || null
    ]);

    const sql = 'INSERT IGNORE INTO Terminal (客户编码, 所属片区, 所属大区) VALUES ?';
    db.query(sql, [values], (err, result) => {
        if (err) {
            console.error('Failed to insert Terminal records:', err);
            res.status(500).json({ success: false, error: err.message });
        } else {
            res.json({ success: true, message: `${result.affectedRows} Terminal records imported successfully (duplicates automatically skipped)` });
        }
    });
});

// ============ Query abnormal time records ============
app.get('/api/audit_visit/search_time/getAbnormalTime', async (req, res) => {
    lastHeartbeat = Date.now();
    clientConnected = true;

    if (!(await ensureDatabase())) {
        return res.status(500).json({ success: false, error: 'Database unavailable, please refresh the page and try again' });
    }

    const db = app.get('db');

    let {
        normalStartTime = '07:00:00',
        normalEndTime = '21:30:00',
        visitor = '',
        customerName = '',
        customerCode = '',
        startDate = '',
        endDate = '',
        area = '',
        region = ''
    } = req.query;

    const abnormalCondition = `(
        TIME(STR_TO_DATE(v.\`拜访开始时间\`, '%Y/%m/%d %H:%i')) < ?
        OR TIME(STR_TO_DATE(v.\`拜访开始时间\`, '%Y/%m/%d %H:%i')) > ?
        OR TIME(STR_TO_DATE(v.\`拜访结束时间\`, '%Y/%m/%d %H:%i')) < ?
        OR TIME(STR_TO_DATE(v.\`拜访结束时间\`, '%Y/%m/%d %H:%i')) > ?
    )`;

    let conditions = [abnormalCondition];
    let params = [normalStartTime, normalEndTime, normalStartTime, normalEndTime];

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
    if (area) {
        conditions.push('t.`所属片区` LIKE ?');
        params.push(`%${area}%`);
    }
    if (region) {
        conditions.push('t.`所属大区` LIKE ?');
        params.push(`%${region}%`);
    }

    const whereClause = conditions.join(' AND ');

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
        ORDER BY v.\`拜访开始时间\` DESC;
    `;

    db.query(sql, params, (err, results) => {
        if (err) {
            console.error('Failed to query abnormal time data:', err);
            res.status(500).json({ success: false, error: err.message });
        } else {
            res.json({ success: true, data: results });
        }
    });
});

// Get all area list
app.get('/api/audit_visit/search_time/getAreas', (req, res) => {
    lastHeartbeat = Date.now();
    const db = app.get('db');
    if (!db || cleanedUp) {
        return res.json({ success: true, data: [] });
    }
    const sql = 'SELECT DISTINCT 所属片区 FROM Terminal WHERE 所属片区 IS NOT NULL AND 所属片区 != "" ORDER BY 所属片区';
    db.query(sql, (err, results) => {
        if (err) {
            res.status(500).json({ success: false, error: err.message });
        } else {
            res.json({ success: true, data: results.map(r => r.所属片区) });
        }
    });
});

// Get all region list
app.get('/api/audit_visit/search_time/getRegions', (req, res) => {
    lastHeartbeat = Date.now();
    const db = app.get('db');
    if (!db || cleanedUp) {
        return res.json({ success: true, data: [] });
    }
    const sql = 'SELECT DISTINCT 所属大区 FROM Terminal WHERE 所属大区 IS NOT NULL AND 所属大区 != "" ORDER BY 所属大区';
    db.query(sql, (err, results) => {
        if (err) {
            res.status(500).json({ success: false, error: err.message });
        } else {
            res.json({ success: true, data: results.map(r => r.所属大区) });
        }
    });
});

// ============ Cleanup: Drop database ============
app.post('/api/audit_visit/search_time/cleanup', async (req, res) => {
    try {
        await dropDatabase();
        res.json({ success: true, message: `Database ${dbName} and all its tables have been deleted` });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Core function to drop database (does NOT exit process)
async function dropDatabase() {
    if (cleanedUp) return;
    cleanedUp = true;
    clientConnected = false;

    const db = app.get('db');
    if (db) {
        try { db.end(); } catch (e) { /* Ignore */ }
        app.set('db', null);
    }

    try {
        const cleanupDb = mysql.createConnection(DB_CONFIG);

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
        console.log(`Database deleted: ${dbName}`);
    } catch (error) {
        console.error('Error dropping database:', error.message);
    }
}

// ============ Heartbeat timeout: only clean database, do NOT exit ============
function setupHeartbeatCheck() {
    setInterval(async () => {
        // Only check timeout if a client has connected and database hasn't been cleaned yet
        if (!clientConnected || cleanedUp) return;

        const elapsed = Date.now() - lastHeartbeat;
        if (elapsed > HEARTBEAT_TIMEOUT) {
            console.log(`\nHeartbeat timeout (${Math.round(elapsed / 1000)}s no activity). User left. Cleaning up database...`);
            try {
                await dropDatabase();
                console.log('Database cleaned up. Server continues running, waiting for next user...');
            } catch (error) {
                console.error('Cleanup error:', error.message);
            }
        }
    }, HEARTBEAT_CHECK_INTERVAL);
}

// Process exit cleanup (Ctrl+C or kill): drop database AND exit
function setupProcessCleanup() {
    async function handleExit(signal) {
        console.log(`\nReceived ${signal}, cleaning up...`);
        try {
            await dropDatabase();
            console.log('Cleanup done, exiting');
        } catch (error) {
            console.error('Cleanup error:', error.message);
        }
        process.exit(0);
    }

    process.on('SIGINT', () => handleExit('SIGINT'));
    process.on('SIGTERM', () => handleExit('SIGTERM'));
}

// ============ Start server ============
initDatabase().then(() => {
    setupProcessCleanup();
    setupHeartbeatCheck();
    app.listen(port, () => {
        console.log(`Server running on http://localhost:${port}`);
        console.log(`Current database: ${dbName}`);
        console.log(`Heartbeat timeout: ${HEARTBEAT_TIMEOUT / 1000}s`);
        console.log('Timeout behavior: clean database only (server keeps running)');
        console.log('Waiting for client connection...');
    });
}).catch(err => {
    console.error('Failed to start:', err);
    process.exit(1);
});
