const express = require('express');
const mysql = require('mysql2');
const cors = require('cors');
const bodyParser = require('body-parser');

const app = express();
const port = 8011;

app.use(bodyParser.json({ limit: '10000mb' }));
app.use(bodyParser.urlencoded({ limit: '10000mb', extended: true }));
app.use(cors());

const dbName = `terminal_${Date.now()}`;

const baseDb = mysql.createConnection({
    host: 'localhost',
    user: 'root',
    password: 'Guoyanjun123.'
});

let lastHeartbeat = Date.now();
let cleanedUp = false;
const HEARTBEAT_TIMEOUT = 90000;
const HEARTBEAT_CHECK_INTERVAL = 15000;

// Initialize database and create Visit and Terminal tables
async function initDatabase() {
    try {
        await new Promise((resolve, reject) => {
            baseDb.connect(err => err ? reject(`Base connection failed: ${err.message}`) : resolve());
        });

        await new Promise((resolve, reject) => {
            baseDb.query(`CREATE DATABASE IF NOT EXISTS \`${dbName}\``, err =>
                err ? reject(`Failed to create database: ${err.message}`) : resolve()
            );
        });

        baseDb.end();

        const db = mysql.createConnection({
            host: 'localhost',
            user: 'root',
            password: 'Guoyanjun123.',
            database: dbName
        });

        await new Promise((resolve, reject) => {
            db.connect(err => err ? reject(`Failed to connect to new database: ${err.message}`) : resolve());
        });

        // Create Visit table with proper indexes
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

        // Create Terminal table (unique customer code to avoid JOIN bloat)
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
        console.log(`Database initialization completed: ${dbName}`);
        console.log('Visit and Terminal tables have been created successfully');

    } catch (error) {
        console.error('Database initialization failed:', error);
        process.exit(1);
    }
}

// ============ Heartbeat endpoint ============
app.get('/api/audit_visit/search_time/heartbeat', (req, res) => {
    lastHeartbeat = Date.now();
    res.json({ success: true, timestamp: lastHeartbeat });
});

// Upload Visit records
app.post('/api/audit_visit/search_time/uploadVisit', (req, res) => {
    console.log('Processing Visit records upload');
    lastHeartbeat = Date.now();
    const db = app.get('db');
    const records = req.body.records;
    console.log(`Number of Visit records to process: ${records.length}`);

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

// Upload Terminal records (use INSERT IGNORE to avoid duplicate customer code errors)
app.post('/api/audit_visit/search_time/uploadTerminal', (req, res) => {
    console.log('Processing Terminal records upload');
    lastHeartbeat = Date.now();
    const db = app.get('db');
    const records = req.body.records;
    console.log(`Number of Terminal records to process: ${records.length}`);

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
// Filters records where visit start/end times fall outside the user-defined normal time range
// User provides a single normal time range (e.g. 07:00 ~ 21:30), any record whose start or end
// time falls outside this range is considered abnormal
app.get('/api/audit_visit/search_time/getAbnormalTime', (req, res) => {
    lastHeartbeat = Date.now();
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

    // Core abnormal time condition:
    // A record is abnormal if its start time OR end time falls outside the normal range [normalStartTime, normalEndTime]
    // Uses STR_TO_DATE to parse the time string (format: YYYY/MM/DD HH:mm) and TIME() to extract just the time part
    const abnormalCondition = `(
        TIME(STR_TO_DATE(v.\`拜访开始时间\`, '%Y/%m/%d %H:%i')) < ?
        OR TIME(STR_TO_DATE(v.\`拜访开始时间\`, '%Y/%m/%d %H:%i')) > ?
        OR TIME(STR_TO_DATE(v.\`拜访结束时间\`, '%Y/%m/%d %H:%i')) < ?
        OR TIME(STR_TO_DATE(v.\`拜访结束时间\`, '%Y/%m/%d %H:%i')) > ?
    )`;

    let conditions = [abnormalCondition];
    let params = [normalStartTime, normalEndTime, normalStartTime, normalEndTime];

    // Additional filter conditions
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

// Get all area list (for frontend dropdown selection)
app.get('/api/audit_visit/search_time/getAreas', (req, res) => {
    lastHeartbeat = Date.now();
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

// Get all region list (for frontend dropdown selection)
app.get('/api/audit_visit/search_time/getRegions', (req, res) => {
    lastHeartbeat = Date.now();
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

// ============ Cleanup: Drop entire database ============
app.post('/api/audit_visit/search_time/cleanup', async (req, res) => {
    try {
        await dropDatabase();
        res.json({ success: true, message: `Database ${dbName} and all its tables have been deleted` });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Core function to drop database
async function dropDatabase() {
    if (cleanedUp) return;
    cleanedUp = true;

    const db = app.get('db');
    if (db) {
        try { db.end(); } catch (e) { /* Ignore */ }
    }

    const cleanupDb = mysql.createConnection({
        host: 'localhost',
        user: 'root',
        password: 'Guoyanjun123.'
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
    console.log(`Database deleted successfully: ${dbName}`);
}

// ============ Heartbeat timeout auto cleanup ============
function setupHeartbeatCheck() {
    const checkTimer = setInterval(async () => {
        const elapsed = Date.now() - lastHeartbeat;
        if (elapsed > HEARTBEAT_TIMEOUT && !cleanedUp) {
            console.log(`\nHeartbeat timeout (no activity for ${Math.round(elapsed / 1000)} seconds), user has left. Automatically cleaning up database...`);
            clearInterval(checkTimer);
            try {
                await dropDatabase();
                console.log('Automatic cleanup completed, server will exit now');
            } catch (error) {
                console.error('Error occurred during automatic cleanup:', error.message);
            }
            process.exit(0);
        }
    }, HEARTBEAT_CHECK_INTERVAL);
}

// Setup process cleanup on exit (Ctrl+C or kill signal)
function setupProcessCleanup() {
    async function handleExit(signal) {
        if (cleanedUp) {
            process.exit(0);
            return;
        }
        console.log(`\nReceived ${signal} signal, cleaning up database before exit...`);
        try {
            await dropDatabase();
            console.log('Cleanup completed, process exiting');
        } catch (error) {
            console.error('Error occurred during cleanup:', error.message);
        }
        process.exit(0);
    }

    process.on('SIGINT', () => handleExit('SIGINT'));
    process.on('SIGTERM', () => handleExit('SIGTERM'));
}

// Initialize database and start server
initDatabase().then(() => {
    setupProcessCleanup();
    setupHeartbeatCheck();
    app.listen(port, () => {
        console.log(`Server running on http://localhost:${port}`);
        console.log(`Current database: ${dbName}`);
        console.log(`Heartbeat timeout: ${HEARTBEAT_TIMEOUT / 1000} seconds (database will be automatically cleaned up and server exited after user closes the page)`);
    });
});
