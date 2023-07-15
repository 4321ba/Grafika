//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : nem adjuk meg a nevunket random weboldalakon
// Neptun : a neptunkodunkat sem
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"
#include <cmath>
#include <cstdio>

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	layout(location = 0) in vec3 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, vp.z + 1);		// modeling space + vec3(0,0,1) = normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;	   // virtual world on the GPU

const vec3 PIROS = vec3(1, 0, 0);
const vec3 ZOLD = vec3(0, 1, 0);
const vec3 KEK = vec3(0, 0, 1);
const vec3 FEKETE = vec3(0, 0, 0);
const vec3 SZURKE = vec3(0.5, 0.5, 0.5);
const vec3 FEHER = vec3(1, 1, 1);

const float SEBESSEG = 1; // koordinataegyseg / sec
const float FORGASSEBESSEG = 2; // rad / sec
const float SZEMELFORGATAS = 2 * M_PI * 30 / 360; // 30 fokkal vannak elforgatva a szemek a szajhoz kepest

const float HAMISUGAR = 0.2; // koordinataegyseg
const float SZEMSUGAR = 0.05;
const float SZEMBELSOSUGAR = 0.03;
const float MAXSZAJSUGAR = 0.05;
float szajSugar = 0;

bool isEPressed = false;
bool isSPressed = false;
bool isFPressed = false;

// muszaj ujraimplementalni, mert a frameworkben nem furiszorzatot hasznal
inline float dotLorentz(const vec3& v1, const vec3& v2) { return (v1.x * v2.x + v1.y * v2.y - v1.z * v2.z); }
inline float lengthLorentz(const vec3& v) { return sqrtf(dotLorentz(v, v)); }
inline vec3 normalizeLorentz(const vec3& v) { return v * (1 / lengthLorentz(v)); }
inline vec3 crossLorentz(const vec3& v1, const vec3& v2) { return cross(vec3(v1.x, v1.y, -v1.z), vec3(v2.x, v2.y, -v2.z)); }

// 'v0' parameter: normalizalt irany, 'v' parameter: nem kell normalizalt legyen

vec3 merolegesIrany(const vec3& p, const vec3& v) {
	return normalizeLorentz(crossLorentz(p, v));
}

vec3 pontAdottIranybaEsTavolsagra(const vec3& p, const vec3& v0, float tav) {
	return p * coshf(tav) + v0 * sinhf(tav);
}

// p pontbol indulva, v0 iranyu, vHossz nagysagu sebesseggel, t ido mulva; visszaadja a parameterkent kapott valtozokat modositva
void menjElore(vec3& p, vec3& v0, float vHossz, float t) {
	float s = vHossz * t;
	vec3 ujP = pontAdottIranybaEsTavolsagra(p, v0, s);
	v0 = -(p - ujP * coshf(s) ) / sinhf(s);
	p = ujP;
}

float ketPontTavolsaga(const vec3& p, const vec3& q) {
	return acoshf(-dotLorentz(p, q));
}

vec3 pontIranyaMasikPontbol(const vec3& innen, const vec3& ennek) {
	float tav = ketPontTavolsaga(innen, ennek);
	return (ennek - innen * coshf(tav)) / sinhf(tav);
}

vec3 vektorElforgatas(const vec3& p, const vec3& v, float szog) {
	return (normalizeLorentz(v)*cosf(szog) + merolegesIrany(p, v)*sinf(szog)) * lengthLorentz(v);
}

void pontVisszavetites(vec3& p) {
	p.z = sqrtf(p.x * p.x + p.y * p.y + 1);
}
void vektorVisszavetites(const vec3& p, vec3& v) {
	float lambda = -dotLorentz(p, v) / dotLorentz(p, p);
	v = v + lambda * p;
}

void korRajzolo(const vec3& pos, float radius, const vec3& color) {
	gpuProgram.setUniform(color, "color");

	enum { ENNYISZOG = 64, CSUCSPONTOKSZAMA = ENNYISZOG + 3 };
	//+1 a kozepe miatt, +1 mert az egyik csucs duplan kell legyen, +1 hogy a kerekitesi hiba miatt se legyen 1 px vekony csik pl ahol osszeer

	vec3 vertices[CSUCSPONTOKSZAMA];
	vertices[0] = pos;

	// adott pontban ervenyes irany/egysegvektor generalasa
	vec3 v0 = vec3(1, 0, 0);
	vektorVisszavetites(pos, v0);
	v0 = normalizeLorentz(v0);

	for (int i = 1; i < CSUCSPONTOKSZAMA; ++i) {
		vertices[i] = pontAdottIranybaEsTavolsagra(pos, v0, radius);
		v0 = vektorElforgatas(pos, v0, M_PI * 2.0 / ENNYISZOG);
	}
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glDrawArrays(GL_TRIANGLE_FAN, 0, CSUCSPONTOKSZAMA);
}

void vonalRajzolo(const std::vector<vec3>& pontok) {
	gpuProgram.setUniform(FEHER, "color");
	glBufferData(GL_ARRAY_BUFFER, pontok.size() * sizeof(vec3), pontok.data(), GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glDrawArrays(GL_LINE_STRIP, 0, pontok.size());
}

struct Hami {
	vec3 pos, v0, color;
	std::vector<vec3> utvonal;

	Hami(vec3 p, vec3 v0, vec3 c): pos(p), v0(v0), color(c) {}

	void kirajzol(const vec3& ideNezzen) {
		korRajzolo(pos, HAMISUGAR, color);
		korRajzolo(pontAdottIranybaEsTavolsagra(pos, v0, HAMISUGAR), szajSugar, FEKETE);

		vec3 balSzemPos = pontAdottIranybaEsTavolsagra(pos, vektorElforgatas(pos, v0, SZEMELFORGATAS), HAMISUGAR);
		vec3 balNezesIranya = pontIranyaMasikPontbol(balSzemPos, ideNezzen);
		korRajzolo(balSzemPos, SZEMSUGAR, FEHER);
		korRajzolo(pontAdottIranybaEsTavolsagra(balSzemPos, balNezesIranya, SZEMSUGAR - SZEMBELSOSUGAR / 2), SZEMBELSOSUGAR, KEK);

		vec3 jobbSzemPos = pontAdottIranybaEsTavolsagra(pos, vektorElforgatas(pos, v0, -SZEMELFORGATAS), HAMISUGAR);
		vec3 jobbNezesIranya = pontIranyaMasikPontbol(jobbSzemPos, ideNezzen);
		korRajzolo(jobbSzemPos, SZEMSUGAR, FEHER);
		korRajzolo(pontAdottIranybaEsTavolsagra(jobbSzemPos, jobbNezesIranya, SZEMSUGAR - SZEMBELSOSUGAR / 2), SZEMBELSOSUGAR, KEK);
	}
	void haladj(double delta) {
		menjElore(pos, v0, SEBESSEG, delta);
		pontVisszavetites(pos);
		vektorVisszavetites(pos, v0);
		v0 = normalizeLorentz(v0);
		utvonal.push_back(pos);
	}
	void fordulj(double delta) {
		v0 = vektorElforgatas(pos, v0, FORGASSEBESSEG * delta);
		vektorVisszavetites(pos, v0);
		v0 = normalizeLorentz(v0);
	}
};

Hami jatekos(vec3(0, 0, 1), vec3(0, 1, 0), PIROS);
Hami korpalyas(vec3(2, 2, 3), vec3(1, -1, 0) / sqrtf(2), ZOLD);

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	glGenVertexArrays(1, &vao);	// get 1 vao id
	glBindVertexArray(vao);		// make it active

	unsigned int vbo;		// vertex buffer object
	glGenBuffers(1, &vbo);	// Generate 1 buffer
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(SZURKE.x, SZURKE.y, SZURKE.z, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer
	glBindVertexArray(vao);  // Draw call

	korRajzolo(vec3(0, 0, 1), 32, FEKETE);
	vonalRajzolo(jatekos.utvonal);
	vonalRajzolo(korpalyas.utvonal);
	jatekos.kirajzol(korpalyas.pos);
	korpalyas.kirajzol(jatekos.pos);

	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'e') isEPressed = true;
	if (key == 's') isSPressed = true;
	if (key == 'f') isFPressed = true;
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
	if (key == 'e') isEPressed = false;
	if (key == 's') isSPressed = false;
	if (key == 'f') isFPressed = false;
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {}

// delta masodpercekben
void frissites(double delta) {
	if (isEPressed)
		jatekos.haladj(delta);
	if (isFPressed)
		jatekos.fordulj(-delta);
	if (isSPressed)
		jatekos.fordulj(delta);

	korpalyas.haladj(delta);
	korpalyas.fordulj(delta);
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time in ms since the start of the program
	static long elozoTime = 0;
	long deltams = time - elozoTime; // legutobbi frissites ota eltelt ido ms-ban
	if (deltams <= 2)  // ha irrealisan kicsi ido telt el, akkor nem frissitunk, kulonben bekrepal (legalabbis 0-nal biztosan), vagy pontatlan lesz (1;2)
		return;
	elozoTime = time;
	
	int updateekSzama = deltams / 50 + 1; // ha tul nagy ugras lenne, akkor tobbszor updateelunk, es ugy teszunk, mintha kb. 50msenkent tortennenek a frissitesek
	for (int i = 0; i < updateekSzama; ++i)
		frissites(deltams / (double)updateekSzama / 1000.0);
	
	szajSugar = sin(time / 1000.0) * sin(time / 1000.0) * MAXSZAJSUGAR;
	
	glutPostRedisplay();
}
