// Exporta los aldeanos con «Ropa Actual» del Wardrobe Studio a .glb (botón «GLB ligero»).
// Uso: node tools/aldeanos/exportar-aldeanos.cjs [salida=personajes/aldeanos] [solo=mod_streetwear,...]
// Abre el estudio en Chromium sin ventana (Playwright), sirve public/assets/aistudio desde este mismo proceso,
// elige cada look de la pestaña «Aldea» y guarda lo que descarga el botón de exportar.
const http = require('http'), fs = require('fs'), path = require('path');
const { chromium } = require('playwright');

const RAIZ = path.resolve(__dirname, '../../public/assets/aistudio');
const PAGINA = 'Parametric_Character_Outfitter___Wardrobe_Studio__10__fc55dc6c.html';
const SALIDA = path.resolve(process.argv[2] || path.resolve(__dirname, '../../personajes/aldeanos'));
const SOLO = process.argv[3] ? process.argv[3].split(',') : null;
// Género de cada look (el estudio lo elige al azar si no): mitad hombres, mitad mujeres
const GENERO = { mod_chic_dress: 'mujer', mod_trench: 'mujer', mod_summer: 'mujer', mod_sport: 'mujer', mod_doctor: 'mujer', mod_college: 'mujer' };
const TIPOS = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.glb': 'model/gltf-binary', '.png': 'image/png', '.jpg': 'image/jpeg', '.webp': 'image/webp' };

const servidor = http.createServer((req, res) => {
  const f = path.join(RAIZ, decodeURIComponent(req.url.split('?')[0]));
  if (!f.startsWith(RAIZ) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { 'Content-Type': TIPOS[path.extname(f)] || 'application/octet-stream' });
  fs.createReadStream(f).pipe(res);
});

(async () => {
  await new Promise(ok => servidor.listen(0, ok));
  const url = `http://127.0.0.1:${servidor.address().port}/${PAGINA}`;
  fs.mkdirSync(SALIDA, { recursive: true });
  const nav = await chromium.launch();
  const p = await nav.newPage({ viewport: { width: 1400, height: 900 }, acceptDownloads: true });
  const errores = []; p.on('pageerror', e => errores.push(e.message));
  await p.goto(url, { waitUntil: 'load', timeout: 120000 });
  await p.waitForFunction(() => document.querySelectorAll('#villager-professions-grid button').length > 0, null, { timeout: 120000 });
  await p.waitForTimeout(3000);
  await p.evaluate(() => document.getElementById('btn-tab-aldeanos').click());
  await p.selectOption('#villager-era-select', 'modern');
  await p.selectOption('#villager-build-select', 'natural');
  const looks = await p.$$eval('#villager-professions-grid button[data-villager-prof]', bs => bs.map(b => ({ id: b.dataset.villagerProf, nombre: b.querySelector('span').textContent })));
  const indice = [];
  for (const look of looks) {
    if (SOLO && !SOLO.includes(look.id)) continue;
    const genero = GENERO[look.id] || 'hombre';
    await p.selectOption('#villager-gender-select', genero);
    await p.click(`#villager-professions-grid button[data-villager-prof="${look.id}"]`);
    await p.waitForTimeout(2500);
    const espera = p.waitForEvent('download', { timeout: 180000 });
    await p.evaluate(() => document.getElementById('btn-export-light-char-glb').click());
    const d = await espera;
    const nombre = look.id.replace(/^mod_/, '') + '.glb';
    await d.saveAs(path.join(SALIDA, nombre));
    await p.screenshot({ path: path.join(SALIDA, look.id.replace(/^mod_/, '') + '.png'), clip: { x: 350, y: 60, width: 700, height: 820 } });
    const kb = Math.round(fs.statSync(path.join(SALIDA, nombre)).size / 1024);
    console.log('✔', nombre, look.nombre, genero, kb + ' KB');
    indice.push({ archivo: nombre, id: look.id, nombre: look.nombre, genero });
  }
  fs.writeFileSync(path.join(SALIDA, 'exportados.json'), JSON.stringify(indice, null, 2));
  if (errores.length) console.log('Errores de la página:\n' + errores.slice(0, 10).join('\n'));
  await nav.close(); servidor.close();
})().catch(e => { console.error(e); process.exit(1); });
