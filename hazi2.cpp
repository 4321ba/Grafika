//=============================================================================================
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

// felhasznalt forras: a targy moodle honlapjan talalhato minimalis sugarkoveto CPU-n demo

struct Hit {
	float t = -1;
	vec3 position, normal;
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 s, vec3 d): start(s), dir(normalize(d)) {}
};

class Intersectable {
public:
	virtual Hit intersect(const Ray& ray) = 0;
	virtual ~Intersectable() {}
};

// felter / sik
class HemiVolume : public Intersectable {
	vec3 normal, point; // normal a terfogatbol kifele mutat
	bool bothSides = true; // mindket iranybol utkozzon-e
public:
	void transform(mat4 m, bool inverz) { // logikailag rossz mert a normalvektort inverz trafozni kene, de a mi celunknak most megfelel
		vec4 n = vec4(normal.x, normal.y, normal.z, 0) * m;
		vec4 p = vec4(point.x, point.y, point.z, 1) * m;
		normal = (inverz ? -1 : 1) * normalize(vec3(n.x, n.y, n.z));
		point = vec3(p.x/p.w, p.y/p.w, p.z/p.w);
		bothSides = !inverz;
	}

	vec4 sik() {
		return vec4(normal.x, normal.y, normal.z, -dot(normal, point));
	}
	HemiVolume(const vec3& n, const vec3& p): normal(normalize(n)), point(p) {}
	HemiVolume(const vec3& p1, const vec3& p2, const vec3& p3): normal(normalize(cross(p2-p1, p3-p1))), point(p1) {}
	Hit intersect(const Ray& ray) {
		Hit hit;
		if (!bothSides && dot(ray.dir, normal) >= 0)
			return hit;
		float t = -1 * dot(sik(), vec4(ray.start.x, ray.start.y, ray.start.z, 1)) / dot(sik(), vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0));
		if (t < 0)
			return hit;
		hit.t = t;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normal;
		return hit;
	}
	bool isInside(const vec3& p) {
		return dot(sik(), vec4(p.x, p.y, p.z, 1)) < 0;
	}
};

// sikokbol felepulo konvex terfogat pl kocka
class ConvexVolume : public Intersectable {
	std::vector<HemiVolume> volumes;
	bool inverz = false; // atlatunk rajta es a belso oldalak latszanak
	
	bool insideAllHemiVolumesExcept(const HemiVolume& v, const vec3& p) {
		for (HemiVolume& vol: volumes)
			if (&v != &vol && inverz == vol.isInside(p))
				return false;
		return true;
	}
	
public:
	ConvexVolume(const std::vector<HemiVolume>& vols, const mat4& transform, bool inverz = false): volumes(vols), inverz(inverz) {
		for (HemiVolume& vol: volumes)
            vol.transform(transform, inverz);
	}
	
	Hit intersect(const Ray& ray) {
		Hit bestHit;
		for (HemiVolume& vol: volumes) {
			Hit thisHit = vol.intersect(ray);
			if (thisHit.t >= 0 && (bestHit.t < 0 || bestHit.t > thisHit.t) && insideAllHemiVolumesExcept(vol, thisHit.position))
				bestHit = thisHit;
		}
		return bestHit;
	}
};

// duplikalt oldalak letorolve
const char *cubeObj = R"(
v  0.0  0.0  0.0
v  0.0  0.0  1.0
v  0.0  1.0  0.0
v  0.0  1.0  1.0
v  1.0  0.0  0.0
v  1.0  0.0  1.0
v  1.0  1.0  0.0
v  1.0  1.0  1.0

f  1  7  5
f  1  4  3
f  3  8  7
f  5  7  8
f  1  5  6
f  2  6  8
)";

// duplikalt oldalak letorolve
const char *dodecahedronObj = R"(
v  -0.57735  -0.57735  0.57735
v  0.934172  0.356822  0
v  0.934172  -0.356822  0
v  -0.934172  0.356822  0
v  -0.934172  -0.356822  0
v  0  0.934172  0.356822
v  0  0.934172  -0.356822
v  0.356822  0  -0.934172
v  -0.356822  0  -0.934172
v  0  -0.934172  -0.356822
v  0  -0.934172  0.356822
v  0.356822  0  0.934172
v  -0.356822  0  0.934172
v  0.57735  0.57735  -0.57735
v  0.57735  0.57735  0.57735
v  -0.57735  0.57735  -0.57735
v  -0.57735  0.57735  0.57735
v  0.57735  -0.57735  -0.57735
v  0.57735  -0.57735  0.57735
v  -0.57735  -0.57735  -0.57735

f  19  3  2
f  8  14  2
f  20  5  4
f  13  17  4
f  7  16  4
f  6  15  2
f  10  18  3
f  11  1  5
f  20  9  8
f  9  16  7
f  12  15  6
f  13  1  11
)";

const char *icosahedronObj = R"(
v  0  -0.525731  0.850651
v  0.850651  0  0.525731
v  0.850651  0  -0.525731
v  -0.850651  0  -0.525731
v  -0.850651  0  0.525731
v  -0.525731  0.850651  0
v  0.525731  0.850651  0
v  0.525731  -0.850651  0
v  -0.525731  -0.850651  0
v  0  -0.525731  -0.850651
v  0  0.525731  -0.850651
v  0  0.525731  0.850651

f  2  3  7
f  2  8  3
f  4  5  6
f  5  4  9
f  7  6  12
f  6  7  11
f  10  11  3
f  11  10  4
f  8  9  10
f  9  8  1
f  12  1  2
f  1  12  5
f  7  3  11
f  2  7  12
f  4  6  11
f  6  5  12
f  3  8  10
f  8  2  1
f  4  10  9
f  5  9  1
)";

// obj vegen \0 char elott egyetlen sortores!
std::vector<HemiVolume> parseObj(const char* obj) {
	std::vector<vec3> coords;
	std::vector<HemiVolume> faces;
	while (*obj != '\0') {
		while (*obj == '\n')
			++obj;
		char buf[3];
		vec3 p;
		sscanf(obj, "%s %f %f %f", buf, &p.x, &p.y, &p.z);
		if (std::string(buf) == "v")
			coords.push_back(p);
		else if (std::string(buf) == "f")
			faces.push_back(HemiVolume(coords[int(p.x-0.5)], coords[int(p.y-0.5)], coords[int(p.z-0.5)]));
		while (*obj++ != '\n')
			;
	}
	return faces;
}


struct Cone : public Intersectable {
	vec3 normal, point, color; // normal egysegvektor
	float alpha, height;

	Cone(const vec3& n, const vec3& p, float a, float h, const vec3& c):
		normal(normalize(n)), point(p), color(c), alpha(a), height(h) {}

	Hit intersect(const Ray& ray) {
		Hit hit;
		const vec3& s = ray.start;
		const vec3& d = ray.dir;
		const vec3& n = normal;
		const vec3& p = point;
		float cos2a = cosf(alpha) * cosf(alpha);
		// f(s+d*t) = (ndt+n(s-p))^2 - (dt+s-p)^2*cos^2a = 0
		// (nd)^2*t^2 + 2nd*n(s-p)*t + (ns-np)^2 - (d^2*t^2 + 2*d*(s-p)*t + (s-p)^2)*cos^2a = 0
		// ((nd)^2-d^2*cos^2a)*t^2 + (2nd*n(s-p)-2*d*(s-p)*cos^2a)*t + (ns-np)^2 - (s-p)^2*cos^2a = 0
		float a = dot(d, n) * dot(d, n) - dot(d, d) * cos2a;
		float b = 2 * dot(n, d) * dot(n, s-p) - 2 * dot(d, s-p) * cos2a;
		float c = dot(n, s-p) * dot(n, s-p) - dot(s-p, s-p) * cos2a;
		float disc = b * b - 4 * a * c;
		if (disc < 0)
			return hit;
		float t1 = (-b - sqrtf(disc)) / 2.0f / a;
		float t2 = (-b + sqrtf(disc)) / 2.0f / a;
		if (t1 < 0 || dot(s+d*t1 - p, n) < 0 || dot(s+d*t1 - p, n) > height)
			t1 = -1;
		if (t2 < 0 || dot(s+d*t2 - p, n) < 0 || dot(s+d*t2 - p, n) > height)
			t2 = -1;
		if (t1 < 0 && t2 < 0)
			return hit;
		if (t1 < 0 || (t2 > 0 && t2 < t1))
			t1 = t2;
		hit.t = t1;
		hit.position = ray.start + ray.dir * hit.t;
		const vec3& r = hit.position;
		hit.normal = normalize(2 * dot(r-p, n) * n - 2 * (r-p) * cos2a);
		return hit;
	}
};

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

const float epsilon = 0.003f; // 0.0001f volt, de szoszos a kup sajat megvilagitasa (pontatlansag)

bool approxEq(const vec3& a, const vec3& b) {
    return fabsf(a.x - b.x) < epsilon && fabsf(a.y - b.y) < epsilon && fabsf(a.z - b.z) < epsilon;
}

class Scene {
	std::vector<Intersectable *> objects;
	std::vector<Cone *> cones;
	Camera camera;
public:
	
	void moveConeTo(int X, int Y) {
		Hit h = firstIntersect(camera.getRay(X, Y));
		if (h.t < 0 || cones.size() == 0)
			return;
		Cone* bestCone = cones[0];
		for (Cone* cone : cones)
			if (length(cone->point - h.position) < length(bestCone->point - h.position))
				bestCone = cone;
		bestCone->point = h.position;
		bestCone->normal = h.normal;
	}
	
	void build() {
		vec3 eye = vec3(0, 0, 2), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);
		
		objects.push_back(new ConvexVolume(parseObj(cubeObj), 
			ScaleMatrix(vec3(1, 1, 1)) *
			RotationMatrix(0.6, vec3(0, 1, 0)) *
			TranslateMatrix(vec3(-0.75, -0.5, 0)), true // atlatszo oldalu kivulrol
		));
		objects.push_back(new ConvexVolume(parseObj(dodecahedronObj), 
			ScaleMatrix(vec3(0.3, 0.3, 0.3)) *
			RotationMatrix(0.4, vec3(0, 1, 0)) *
			TranslateMatrix(vec3(-0.3, -0.25, 0))
		));
		objects.push_back(new ConvexVolume(parseObj(icosahedronObj), 
			ScaleMatrix(vec3(0.3, 0.3, 0.3)) *
			RotationMatrix(0.2, vec3(0, 1, 0)) *
			TranslateMatrix(vec3(0.2, -0.25, 0.1))
		));
		
		Cone* c1 = new Cone(vec3(0.564642, -0.000000, 0.825336), vec3(-0.154560, 0.337373, -0.407362), 0.5, 0.1, vec3(1,0,0));
		Cone* c2 = new Cone(vec3(-0.000000, -1.000000, -0.000000), vec3(0.180912, 0.500000, -0.063431), 0.5, 0.1, vec3(0,1,0));
		Cone* c3 = new Cone(vec3(0.331259, 0.525731, 0.783501), vec3(-0.163092, -0.091177, 0.139816), 0.5, 0.1, vec3(0,0,1));
		objects.push_back(c1);
		objects.push_back(c2);
		objects.push_back(c3);
		cones.push_back(c1);
		cones.push_back(c2);
		cones.push_back(c3);
	}

	void render(std::vector<vec4>& image) {
		for (unsigned Y = 0; Y < windowHeight; Y++) {
			for (unsigned X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable * object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))
                bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0)
            bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}
	
	vec3 trace(Ray ray) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return vec3(0, 0, 0); // hatterszin
		float firstColor = 0.2 * (1 + dot(hit.normal, -ray.dir));
		vec3 outRadiance = vec3(firstColor, firstColor, firstColor);
		for (Cone * cone : cones) {
			vec3 shraystart = cone->point + cone->normal * epsilon;
			Ray shadowRay(shraystart, hit.position - shraystart);
			float cosTheta = dot(hit.normal, shadowRay.dir);
			Hit shadowHit = firstIntersect(shadowRay);
			bool shadowIntersect = shadowHit.t > 0 && !approxEq(hit.position, shadowHit.position) && !approxEq(hit.normal, shadowHit.normal);
			if (cosTheta < 0 && !shadowIntersect)
				outRadiance = outRadiance + cone->color * 0.7 * (shadowHit.t < 1.5f ? (1.5f-shadowHit.t)/1.5f : 0.0f);
		}
		return outRadiance;
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

Texture* texture = nullptr;

void rerender() {
	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %ld milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	delete texture;
	texture = new Texture(windowWidth, windowHeight, image);
	glutPostRedisplay();
}

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
    
    unsigned int vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    unsigned int vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    
    float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 }; // two triangles forming a quad
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
    
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    
	scene.build();
    
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");

	rerender();
}

// Window has become invalid: Redraw
void onDisplay() {
    gpuProgram.setUniform(*texture, "textureUnit");
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glutSwapBuffers();
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (state != GLUT_DOWN || button != GLUT_LEFT_BUTTON)
		return;
	scene.moveConeTo(pX, windowHeight - pY); // megforditas
	rerender();
}

void onKeyboard(unsigned char key, int pX, int pY) {}
void onKeyboardUp(unsigned char key, int pX, int pY) {}
void onMouseMotion(int pX, int pY) {}
void onIdle() {}
