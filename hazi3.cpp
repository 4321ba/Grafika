//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
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

// kodokat a moodle-ben talalhato peldakodokbol (3dendzsikebol foleg), es az eloadasdiakbol vettem at

//---------------------------
template<class T> struct Dnum { // Dual numbers for automatic derivation
//---------------------------
	float f; // function value
	T d;  // derivatives
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
};

// Elementary functions prepared for the chain rule as well
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f)*g.d); }

typedef Dnum<vec2> Dnum2;

const int tessellationLevel = 100;

//---------------------------
struct Camera { // 3D camera
//---------------------------
	vec3 wEye, wLookat, wVup;   // extrinsic
	float fov, asp, fp, bp;		// intrinsic
public:
	Camera() {
		asp = (float)windowWidth / windowHeight / 2.0;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 40;
	} 
	mat4 V() { // view matrix: translates the center to the origin
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			                                       u.y, v.y, w.y, 0,
			                                       u.z, v.z, w.z, 0,
			                                       0,   0,   0,   1);
	}

	mat4 P() { // projection matrix
		return mat4(1 / (tan(fov / 2)*asp), 0,                0,                      0,
			        0,                      1 / tan(fov / 2), 0,                      0,
			        0,                      0,                -(fp + bp) / (bp - fp), -1,
			        0,                      0,                -2 * fp*bp / (bp - fp),  0);
	}
};

//---------------------------
struct Material {
//---------------------------
	vec3 kd, ks, ka;
	float shininess;
};

//---------------------------
struct Light {
//---------------------------
	vec3 La, Le;
	vec4 wLightPos; // homogeneous coordinates, can be at ideal point
};

//---------------------------
struct RenderState {
//---------------------------
	mat4	           MVP, M, Minv, V, P;
	Material *         material;
	std::vector<Light> lights;
	Texture *          texture;
	vec3	           wEye;
};

//---------------------------
class Shader : public GPUProgram {
//---------------------------
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};

//---------------------------
class PhongShader : public Shader {
//---------------------------
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec2 texcoord;
		
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La + 
                           (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

//---------------------------
class Geometry {
//---------------------------
protected:
	unsigned int vao, vbo;        // vertex array object
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() = 0;
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

//---------------------------
class ParamSurface : public Geometry {
//---------------------------
	struct VertexData {
		vec3 position, normal;
		vec2 texcoord;
	};

	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }

	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	VertexData GenVertexData(float u, float v) {
		VertexData vtxData;
		vtxData.texcoord = vec2(u, v);
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		vtxData.normal = cross(drdU, drdV);
		return vtxData;
	}

	void create(int N = tessellationLevel, int M = tessellationLevel) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;	// vertices on the CPU
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i *  nVtxPerStrip, nVtxPerStrip);
	}
};

//---------------------------
class Cube : public Geometry {
//---------------------------
	struct VertexData {
		vec3 position, normal;
		vec2 texcoord;
	};

public:
	Cube() {
		// cube.obj-bol
		vec3 vertices[] = {
			vec3(0, 0, 0),
			vec3(0, 0, 1),
			vec3(0, 1, 0),
			vec3(0, 1, 1),
			vec3(1, 0, 0),
			vec3(1, 0, 1),
			vec3(1, 1, 0),
			vec3(1, 1, 1),
		};
		vec3 normals[] = {
			vec3( 0, 0, 1),
			vec3( 0, 0,-1),
			vec3( 0, 1, 0),
			vec3( 0,-1, 0),
			vec3( 1, 0, 0),
			vec3(-1, 0, 0),
		};
		int triangleVtxIdxes[] = {
			1,7,5, 1,3,7, 1,4,3, 1,2,4, 3,8,7, 3,4,8, 5,7,8, 5,8,6, 1,5,6, 1,6,2, 2,6,8, 2,8,4,
		};
		int normalIdxes[] = {
			2,6,3,5,4,1,
		};
		VertexData vtxData[36];	// vertices on the CPU
		for (int i = 0; i < 36; ++i) {
			vtxData[i].position = vertices[triangleVtxIdxes[i] - 1] - vec3(0.5, 0.5, 0.5);
			vtxData[i].normal = normals[normalIdxes[i / 6] - 1];
			vtxData[i].texcoord = vec2(0.5, 0.5); // egyszinu es a textura kozeperol veszi a szint
		}
		
		glBufferData(GL_ARRAY_BUFFER, sizeof(vtxData), vtxData, GL_STATIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw() {
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, 36);
	}
};

float rnd() { return rand() / (float)RAND_MAX; }
//---------------------------
class Terep : public ParamSurface {
//---------------------------
	static const int MAXF = 6;
	float A[MAXF][MAXF]; // ez mar 1/f-szerese lesz az eredetinek
	float fi[MAXF][MAXF];
public:
	Terep() {
		for (int f1 = 0; f1 < MAXF; ++f1) {
			for (int f2 = 0; f2 < MAXF; ++f2) {
				fi[f1][f2] = rnd() * M_PI * 2;
				A[f1][f2] = (f1+f2 == 0) ? 0 : 0.1 * rnd() / sqrtf(f1*f1 + f2*f2);
			}
		}
		create();
	}
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		X = U - 0.5;
		Z = V - 0.5;
		U = U * 2 * M_PI; 
		V = V * 2 * M_PI; 
		for (int f1 = 0; f1 < MAXF; ++f1)
			for (int f2 = 0; f2 < MAXF; ++f2)
				Y = Y + Cos(U * f1 + V * f2 + fi[f1][f2]) * A[f1][f2];
	}
	// szebb lett volna a shaderben megoldani y koord fuggvenyeben, de most jo lesz igy is
	Texture* genTexture() {
		std::vector<vec4> buffer;
		for (int i = 0; i < tessellationLevel; i++) {
			for (int j = 0; j < tessellationLevel; j++) {
				Dnum2 X, Y, Z;
				Dnum2 U((j+0.5) / tessellationLevel, vec2(1, 0));
				Dnum2 V((i+0.5) / tessellationLevel, vec2(0, 1));
				eval(U, V, X, Y, Z);
				float t = (Y.f + 0.4) * 1.8;
				t = std::max(0.0f, std::min(1.0f, t));
				static const vec4 brown(0.32, 0.22, 0.1, 1);
				static const vec4 green(0.15, 0.47, 0.14, 1);
				buffer.push_back(t * brown + (1-t) * green);
			}
		}
		return new Texture(tessellationLevel, tessellationLevel, buffer);
	}
};

//---------------------------
struct Object {
//---------------------------
	Shader *   shader;
	Material * material;
	Texture *  texture;
	Geometry * geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
public:
	Object(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}

	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	void Draw(RenderState state) {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void Animate(float tstart, float tend) { }
};

const vec3 oldalhosszak = vec3(0.3, 1, 0.6); // a, b, c oldalhosszak
const vec3 g = vec3(0, -9.8, 0); // g nehezsegi gyorsulas
const vec3 startpos = vec3(0, 0, 0); // kotel teteje, es kezdopozicio
const float m = 60; // tomeg
const float D = 400; // rugoallando
const float l0 = 3; // nyugalmi hossz
const float kozeg = 3; // kozegellenallasi tenyezo
const float theta = m * (oldalhosszak.x * oldalhosszak.x + oldalhosszak.y * oldalhosszak.y) / 12; // tehetetlensegi nyomatek
const float kforgas = 3; // kozegellenallasi tenyezo a forgasra

struct Ugro : public Object {
	// minden mertekegyseg SI: meter, masodperc, kg, stb...
	vec3 sebesseg = vec3(rnd(), 0, 0); // v0 veletlen kezdosebesseg
	vec3 szogsebesseg = vec3(0, 4, 0); // omega
	Ugro(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry) :
		Object(_shader, _material, _texture, _geometry) {
		scale = oldalhosszak;
		translation = startpos;
	}
	
	void Animate(float tstart, float tend) {
		float delta = tend - tstart;
		
		// erok, sebesseg, pozicio szamitasa:
		// rugoero:
		float megnyulas = length(translation - startpos) - l0;
		vec3 K = megnyulas < 0 ? 0 : D * megnyulas * normalize(startpos - translation);
		// kozegellenallas:
		vec3 Fk = -kozeg * sebesseg;
		vec3 Fe = K + m*g + Fk;
		vec3 gyorsulas = Fe / m;
		
		// forgatonyomatek, szogsebesseg, elfordulas szamitasa:
		// kotelero forgatonyomateka:
		vec3 Mk = cross(kcsp() - translation, K);
		// kozegellenallasi forg.nyom.
		vec3 Mkozeg = -kforgas * szogsebesseg;
		// eredo forgatonyomatek
		vec3 Me = Mk + Mkozeg;
		vec3 beta = Me / theta; // szoggyorsulas
		
		szogsebesseg = szogsebesseg + beta * delta;
		// itt kihasznaljuk hogy igazabol Me, beta es omega z tengellyel parhuzamosak
		float sgn = szogsebesseg.z > 0 ? 1 : -1;
		rotationAngle = rotationAngle + sgn * length(szogsebesseg) * delta;
		
		sebesseg = sebesseg + gyorsulas * delta;
		translation = translation + sebesseg * delta;
	}
	void reset() {
		*this = Ugro(shader, material, texture, geometry);
	}
	// kotelcsatlakozas pozicioja:
	vec3 kcsp() {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		vec4 kcsp = vec4(0,-.5,0,1) * M;
		return vec3(kcsp.x, kcsp.y, kcsp.z);
	}
};

Ugro* ugro;
bool started = false; 
vec3 cam1eyepos = vec3(8, 0, -4);

//---------------------------
class Scene {
//---------------------------
	std::vector<Object *> objects;
	std::vector<Light> lights;
public:
	void Build() {
		// Shaders
		Shader * phongShader = new PhongShader();
		
		// Materials
		Material * mat = new Material;
		mat->kd = vec3(1,1,1);
		mat->ks = vec3(0.04,0.04,0.04);
		mat->ka = vec3(0.7f, 0.7f, 0.7f);
		mat->shininess = 20;

		// Geometries
		Geometry * teglatest = new Cube();
		Terep * terep = new Terep();
		
		// Textures
		Texture * terepTexture = terep->genTexture();
		std::vector<vec4> piros;
		for (int i = 0; i < 9; ++i)
			piros.push_back(vec4(0.54, 0.22, 0.09, 1));
		Texture * ugroTexture = new Texture(3, 3, piros);

		// Create objects by setting up their vertex data on the GPU
		ugro = new Ugro(phongShader, mat, ugroTexture, teglatest);
		objects.push_back(ugro);

		Object * padlo = new Object(phongShader, mat, terepTexture, terep);
		padlo->translation = vec3(0, -12, 0);
		padlo->scale = vec3(15, 15, 15);
		objects.push_back(padlo);

		// Lights
		lights.resize(1);
		lights[0].wLightPos = vec4(5, 5, 4, 0);	// ideal point -> directional light source
		lights[0].La = vec3(.8,.8,.8);
		lights[0].Le = vec3(3,3,3);
	}

	void Render(Camera camera) {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		for (Object * obj : objects) obj->Draw(state);
	}

	void Animate(float tstart, float tend) {
		if (!started)
			return;
		for (Object * obj : objects) obj->Animate(tstart, tend);
	}
};

Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
	srand(time(NULL));
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	Camera camera;
	
	glViewport(0, 0, windowWidth/2, windowHeight);
	vec3 lookdir = -normalize(ugro->kcsp() - ugro->translation);
	camera.wEye = ugro->translation;
	camera.wLookat = ugro->translation + lookdir;
	camera.wVup = vec3(0,0,1);//cross(vec3(0, 0, 1), lookdir); // szerintem igy jobban nezne ki
	scene.Render(camera);
	
	glViewport(windowWidth/2, 0, windowWidth/2, windowHeight);
	camera.wEye = cam1eyepos;
	camera.wLookat = vec3(0, -4, 0);
	camera.wVup = vec3(0, 1, 0);
	scene.Render(camera);
	
	glutSwapBuffers();									// exchange the two buffers
}
// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	ugro->reset();
	started = true;
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { }

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	static float tend = 0;
	const float dt = 0.1f; // dt is ”infinitesimal”
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		
		float rotangl = (tend - tstart) * 0.7/*rad/s*/;
		vec4 v = vec4(cam1eyepos.x, cam1eyepos.y, cam1eyepos.z, 0) * RotationMatrix(rotangl, vec3(0, 1, 0));
		cam1eyepos = vec3(v.x, v.y, v.z); // illene akar normalizalni es az eredeti tavval beszorozni, de annyira nem szamit a kicsi pontatlansag most
		
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();
}
