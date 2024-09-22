/**
 * BASIC 2D FEM on a 3D MESH
 * 
 * Solves the Vector Heat Equation PDE with dirichlet boundary conditions:
 * 
 * We follow the method described "The Vector Heat Method", Sharp, Soliman and Crane, 2019
 * The goal is to find the parallel transport over the mesh given vector boundary conditions.
 * 
 * Connection-Laplacian X(x,y, t) = dX(x,y, t)/dt   in the region D
 *           X(x,y, t) = G(x,y)             on the region boundary #D for all times
 *           X(x,y, 0) = X_0(x,y)           initial condition at time t = 0
 * 
 * Where X(x,y, t) is a tangent vector field.
 * 
 * Tangent vectors are encoded at mesh vertices using the "polar map" using complex numbers. 
 * 
 * Pick a vertex vj around vertex vi and define unit vector Xij, then you can define any vector at vertex vi as:
 * 
 * Vi = e^theta_ij Xij
 * 
 * Where theta_ij is an angle [0; 2PI]
 * 
 * The vector Xij is called the basis vector of the local tangent space centered at vi.
 * 
 * We pick any edge as basis vector Xij, however in a curved mesh the angles around vertex vi don't addup to 2PI.
 * 
 * The solution is to rescale the angles to addup to 2PI. Given the angles alpha_ij measured between outward edges 
 * from vertex vi measured in the mesh, we compute rescaled angles theta_ij as:
 * 
 * scale_i = 2PI / (sum_j alpha_ij)
 * 
 * theta_ij = alpha_ij * scale_i
 * 
 * PARALLEL TRANSPORT
 * 
 * P3
 * º            ^ V1
 * |    \     /
 * |      /    \
 * |  /              \
 * º——————-----------º
 * P1               P2
 * |                /
 * |          /
 * |   /
 * º 
 * P4
 * 
 * Lets say P1 in the picture above is the center point of tangent plane and X12 is the basis vector.
 * 
 * Then 
 * P13 = e^theta13 X12
 * P14 = e^theta14 X12
 * 
 * Any vector V1 = e^theta X12 can be parallel transported from P1 to P2, P3 and P4 in the following way:
 * 
 * Parallel transport to P4:
 * 
 * Lets first parallel transport the basis X12 to P4. Suppose the basis vector of P4 tangent plane is X42, then
 * 
 * P14 = e^theta14 X12
 * P41 = e^theta41 X42
 * 
 * Note that P14 = -P41
 * 
 * So 
 * 
 * P14 = -e^theta41 X42
 * 
 * But e^PI = cos PI + i sin PI = -1
 * 
 * P14 = e^PI e^theta41 X42
 * P14 = e^(PI + theta41) X42
 * 
 * Since P14 = -P41 we have:
 * 
 * e^theta14 X12 = e^(PI + theta41) X42
 * 
 * X12 = e^-theta14 e^(PI + theta41) X42
 * 
 * X12 = e^(PI + theta41 - theta14) X42
 * 
 * Lets call r14 = e^(PI + theta41 - theta14) the CONNECTION COHEFFICIENT from P1 to P4 so:
 * 
 * X12 = r14 X42
 * 
 * Having that in mind, the parallel transport of any vector V1 = e^theta X12 from P1 to P4 is:
 * 
 * V1 = e^theta X12
 * V1 = e^theta r14 X42
 * 
 * 
 * 
 * DERIVATION OF CONNECTION LAPLACIAN FROM “VECTOR” DIRICHLET ENERGY
 * 
 * Scalar Dirichlet Energy
 * 
 * E(P) = 1/2 Sum_ij w_ij | F(P_i) - F(P_j) |^2
 * 
 * Where w_ij are cotan weights and F(P) is scalar valued F : R^n -> R
 * 
 * | F(P_i) - F(P_j) |^2 = F(P_i)^2 - 2 F(P_i) F(P_j) + F(P_j)^2
 * D/F_i = 2 F(P_i) - 2 F(P_j)
 * D/F_j = 2 F(P_j) - 2 F(P_i)
 * 
 * dE(P) = Sum_ij w_ij [F(P_i) - F(P_j), F(P_j) - F(P_i)] = [0, 0]
 * 
 * | w_ij         -w_ij |   | F(P_i) |  =  w_ij ( F(P_i) - F(P_j) )
 * | -w_ij         w_ij |   | F(P_j) |     w_ij ( F(P_j) - F(P_i) )
 * 
 * Which is the Laplace operator
 * 
 * Vector Dirichlet Energy
 * 
 * E(P) = Sum_ij w_ij | P_i - R_ij P_j |^2
 * 
 * Where P(X_i) is vector valued P : R^n -> R^m.
 * Norm |.| is the Euclidean norm
 * R_ij is a 2x2 matrix connection coefficient (rotation matrix)
 * 
 * | P_i - R_ij Pj |^2 = (P_i - R_ij Pj)^T (P_i - R_ij P_j)
 * | P_i - R_ij Pj |^2 = (P_i^T - (P_j^T R_ij^T)) (P_i - R_ij P_j)
 * | P_i - R_ij Pj |^2 = P_i^T (P_i - R_ij P_j) - (P_j^T R_ij^T)(P_i - R_ij P_j) 
 * | P_i - R_ij Pj |^2 =  P_i^T P_i - P_i^T R_ij P_j - P_j^T R_ij^T P_i + P_j^T R_ij^T R_ij P_j
 * 
 * Since:
 * * R_ij^T R_ij = 1
 * * P_j^T R_ij^T P_i = P_i^T R_ij P_j
 * 
 * | P_i - R_ij Pj |^2 =  P_i^T P_i - 2 P_i^T R_ij P_j + P_j^T  P_j
 * 
 * Applying the Gradient/Covariant Derivative:
 * 
 * D/P_i | P_i - R_ij Pj |^2 = D/P_i (P_i^T P_i - 2 P_i^T R_ij P_j + P_j^T  P_j)
 * 
 * D/P_i P_i^T P_i - 2 D/P_i P_i^T R_ij P_j + D/P_i P_j^T  P_j
 * 
 * D/P_i P_i^T P_i = 2 P_i^T
 * D/P_i (- 2 P_i^T R_ij P_j) = -2 P_j^T R_ij^T
 * D/P_i P_j^T  P_j = 0
 * 
 * D/P_i E(P) = 2 (P_i^T  - P_j^T R_ij^T)
 * 
 * D/P_j P_i^T P_i = 0
 * D/P_j (- 2 P_i^T R_ij P_j) = -2 P_i^T R_ij
 * D/P_j P_j^T  P_j = 2 P_j^T
 * 
 * D/P_j E(P) = 2 (P_j^T - P_i^T R_ij)
 * 
 * Linear system to solve is:
 * 2 w_ij (P_i^T  - P_j^T R_ij^T) = 0
 * 2 w_ij (P_j^T - P_i^T R_ij) = 0
 * 
 * | w_ij         -R_ij w_ij|    | P_i |  =  w_ij (P_i - R_ij P_j)
 * | -R_ij^T w_ij       w_ij|    | P_j |     w_ij (P_j - R_ij^T P_i)
 * 
 * Which is the Connection-Laplacian
 */
#ifdef WIN32
#define NOMINMAX
#include <windows.h>
#endif

#if defined (__APPLE__) || defined (OSX)
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glut.h>
#endif

#include "GA/c3ga.h"
#include "GA/c3ga_util.h"
#include "GA/gl_util.h"

#include "primitivedraw.h"
#include "gahelper.h"
#include "Laplacian.h"

#include <memory>

#include <vector>
#include <queue>
#include <map>
#include <fstream>
#include <functional>
#include <complex>
#include "numerics.h"
#include "HalfEdge/Mesh.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Geometry>

// #include <ppl.h>

const char *WINDOW_TITLE = "FEM BASIC 2D";

// GLUT state information
int g_viewportWidth = 800;
int g_viewportHeight = 600;

void display();
void reshape(GLint width, GLint height);
void MouseButton(int button, int state, int x, int y);
void MouseMotion(int x, int y);
void KeyboardUpFunc(unsigned char key, int x, int y);
void SpecialFunc(int key, int x, int y);
void SpecialUpFunc(int key, int x, int y);
void Idle();
void DestroyWindow();
Eigen::Vector3d valueToColor( double d );

//using namespace boost;
using namespace c3ga;
using namespace std;
using namespace numerics;

class Camera
{
public:
	float		pos[3];
	float		fw[3];
	float		up[3];
	float		translateVel;
	float		rotateVel;

	Camera()
	{
		float		_pos[] = { 0, 0, 2};
		float		_fw[] = { 0, 0, -1 };
		float		_up[] = { 0, 1, 0 };

		translateVel = 0.005;
		rotateVel = 0.005;
		memcpy(pos, _pos, sizeof(float)*3);
		memcpy(fw, _fw, sizeof(float)*3);
		memcpy(up, _up, sizeof(float)*3);
	}

	void glLookAt()
	{
		gluLookAt( pos[0], pos[1], pos[2], fw[0],  fw[1],  fw[2], up[0],  up[1],  up[2] );
	}
};

class VertexBuffer
{
public:
	std::vector<Eigen::Vector3d> positions; //mesh vertex positions
	std::vector<Eigen::Vector3d> normals; //for rendering (lighting)
	std::vector<Eigen::Vector3d> colors; //for rendering (visual representation of values)
	int size;

	VertexBuffer() : size(0)
	{
	}

	void resize(int size)
	{
		this->size = size;
		positions.resize(size);
		normals.resize(size);
		colors.resize(size);
	}
	int get_size() { return size; }

};

class IndexBuffer {
public:
	std::vector<int> faces;
	int size;

	IndexBuffer() : size(0)
	{
	}

	void resize(int size)
	{
		this->size = size;
		faces.resize(size);
	}
	int get_size() { return size; }

};

Camera g_camera;
Mesh mesh;
vectorE3GA g_prevMousePos;
bool g_rotateModel = false;
bool g_rotateModelOutOfPlane = false;
rotor g_modelRotor = _rotor(1.0);
float g_dragDistance = -1.0f;
int g_dragObject;
bool g_showWires = true;


VertexBuffer vertexBuffer;
IndexBuffer indexBuffer;
std::vector<std::unordered_map<int, double>> corner_angles;
std::vector<std::unordered_map<int, double>> vertex_to_edge_angles;
std::vector<std::unordered_map<int, std::complex<double>>> vertex_to_vertex_rot;
std::shared_ptr<SparseMatrix<std::complex<double>>> A;
Eigen::SparseLU<Eigen::SparseMatrix<std::complex<double>>> solver;
std::set<int> allconstraints;
Eigen::VectorXcd right_hand_side;
Eigen::VectorXcd solutionU;

/// Project u on the orthgonal of n
/// \param u vector to project
/// \param n vector to build orthogonal space from
/// \return projected vector
static Eigen::Vector3d project(const Eigen::Vector3d & u, const Eigen::Vector3d & n)
{
	return u - (u.dot(n) / n.squaredNorm()) * n;
}

void computeCornerAngles(
	Mesh* mesh, 
	std::vector<std::unordered_map<int, double>> &corner_angles
) {
	for(Vertex &vi : mesh->getVertices()) {
		int i = vi.ID;
		Eigen::Vector3d& pi = vi.p;
		corner_angles[i][i] = 0.0;
		int j = -1;
		for(Vertex::EdgeAroundIteratorCCW edgeAroundIter = vi.iteratorCCW() ; !edgeAroundIter.end() ; edgeAroundIter++)
		{
			int k = edgeAroundIter.edge_out()->pair->vertex->ID;
			if(j >= 0) {
				Eigen::Vector3d& pj = mesh->vertexAt(j).p;
				Eigen::Vector3d& pk = mesh->vertexAt(k).p;
				double corner_angle = atan2((pj - pi).cross(pk - pi).norm(), (pj - pi).dot(pk - pi));
				corner_angles[i][j] = corner_angle; // (i, j) angle between edges j and k in CCW order
				corner_angles[i][i] += corner_angle; // (i, i) sum of angles around i
			}
			j = k;
		}
		{
			int k = vi.edge->pair->vertex->ID;
			Eigen::Vector3d& pj = mesh->vertexAt(j).p;
			Eigen::Vector3d& pk = mesh->vertexAt(k).p;
			double corner_angle = atan2((pj - pi).cross(pk - pi).norm(), (pj - pi).dot(pk - pi));
			corner_angles[i][j] = corner_angle; // (i, j) angle between edges j and k in CCW order
			corner_angles[i][i] += corner_angle; // (i, i) sum of angles around i			
		}
		corner_angles[i][i] = 2 * M_PI / corner_angles[i][i]; // s_i = 2 PI / Total_i
	}
}

void computeVertexToEdgeAngles(
	Mesh* mesh, 
	std::vector<std::unordered_map<int, double>> &corner_angles,
	std::vector<std::unordered_map<int, double>> &vertex_to_edge_angles
) {	
	for(Vertex &vi : mesh->getVertices()) {
		int i = vi.ID;
		double edge_angle = 0;
		for(Vertex::EdgeAroundIteratorCCW edgeAroundIter = vi.iteratorCCW() ; !edgeAroundIter.end() ; edgeAroundIter++)
		{
			int j = edgeAroundIter.edge_out()->pair->vertex->ID;
			vertex_to_edge_angles[i][j] = edge_angle ;
			edge_angle += corner_angles[i][j] * corner_angles[i][i];
		}
	}
}

void computeVertexToVertexRotations(
	Mesh* mesh, 
	std::vector<std::unordered_map<int, double>> &vertex_to_edge_angles,
	std::vector<std::unordered_map<int, complex<double>>> &vertex_to_vertex_rot
) {	

	for(Vertex &vi : mesh->getVertices()) {
		int i = vi.ID;
		for(Vertex::EdgeAroundIteratorCCW edgeAroundIter = vi.iteratorCCW() ; !edgeAroundIter.end() ; edgeAroundIter++)
		{
			int j = edgeAroundIter.edge_out()->pair->vertex->ID;
			double vertex_to_vertex_angle = M_PI + vertex_to_edge_angles[j][i] - vertex_to_edge_angles[i][j];
			vertex_to_vertex_rot[i][j] = std::complex<double>(cos(vertex_to_vertex_angle), sin(vertex_to_vertex_angle));
		}
	}
}

double computeEdgeLengths(
	Mesh* mesh
) {
	int count = 0;
	double sumEdgeLengths = 0.0;
	for(auto &eij : mesh->getEdges()) {
		Eigen::Vector3d& pi = eij->vertex->p;
		Eigen::Vector3d& pj = eij->pair->vertex->p;
		sumEdgeLengths += (pj - pi).norm();
		count++;
	}
	return sumEdgeLengths / (double)count;
}

/**
 * Extend computation of per-element stiffness matrix of triangle elements embedded in 3D space.
 * Original method only works for triangle elements on 2D space, see:
 * 
 * Francisco-Javier Sayas, "A gentle introduction to the Finite Element Method", 2015
 */
Eigen::Matrix3cd AssembleStiffnessElementEmbedded(Vertex* v[3], std::vector<std::unordered_map<int, std::complex<double>>> vertex_to_vertex_rot) {

	Eigen::MatrixXd B(3, 2);
	Eigen::MatrixXd Binv(3, 2);
	Eigen::Vector3d gradN[3];
	Eigen::Matrix3cd elementMatrix;

	B(0,0) = v[1]->p.x() - v[0]->p.x(); B(0,1) = v[2]->p.x() - v[0]->p.x();
	B(1,0) = v[1]->p.y() - v[0]->p.y(); B(1,1) = v[2]->p.y() - v[0]->p.y();
	B(2,0) = v[1]->p.z() - v[0]->p.z(); B(2,1) = v[2]->p.z() - v[0]->p.z();
    
	Binv = ((B.transpose() * B).inverse() * B.transpose()).transpose();

	double faceArea = 0.5 * ((v[1]->p - v[0]->p).cross(v[2]->p - v[0]->p)).norm();

	//grad N^k_1(X) = B^-T (-1, -1)
	//grad N^k_2(X) = B^-T (1, 0)
	//grad N^k_3(X) = B^-T (0, 1)

	gradN[0] = Eigen::Vector3d(-Binv(0,0) - Binv(0,1), -Binv(1,0) - Binv(1,1), -Binv(2,0) - Binv(2,1));
	gradN[1] = Eigen::Vector3d( Binv(0,0), Binv(1,0), Binv(2,0));
	gradN[2] = Eigen::Vector3d( Binv(0,1), Binv(1,1), Binv(2,1));
	for( int i = 0 ; i < 3 ; ++i ) { // for each test function
		for (int j = 0 ; j < 3 ; ++j ) { // for each shape function
			if (i < j) continue; // since stifness matrix is symmetric
			//w_ij = area K <grad N^k_i(X), grad N^k_j(X)>
			if(i == j) {
				elementMatrix(i, j) = faceArea * gradN[i].dot(gradN[j]);
			}
			else {
				int ii = v[i]->ID;
				int jj = v[j]->ID;
				elementMatrix(i, j) = faceArea * gradN[i].dot(gradN[j]) * vertex_to_vertex_rot[jj][ii];
				elementMatrix(j, i) = faceArea * gradN[j].dot(gradN[i]) * vertex_to_vertex_rot[ii][jj];
			}
		}
	}
	return elementMatrix;
}

/**
 * Assemble the stiffness matrix. It does not take into account boundary conditions.
 * Boundary conditions will be applied when linear system is pre-factored (LU decomposition)
 * Original method can be found in:
 * 
 * Francisco-Javier Sayas, "A gentle introduction to the Finite Element Method", 2015
 */
std::shared_ptr<SparseMatrix<complex<double>>> AssembleMatrix(Mesh *mesh, std::vector<std::unordered_map<int, std::complex<double>>> vertex_to_vertex_rot, double delta_t) {
	std::shared_ptr<SparseMatrix<complex<double>>> A(new SparseMatrix<complex<double>>(mesh->numVertices(), mesh->numVertices()));
	Eigen::Matrix3cd stiffnessMatrix, massMatrix;
	complex<double> wij, wji;
	Vertex* v[3];
	for (Face& face : mesh->getFaces()) {
		v[0] = face.edge->vertex;
		v[1] = face.edge->next->vertex;
		v[2] = face.edge->next->next->vertex;
		stiffnessMatrix = AssembleStiffnessElementEmbedded(v, vertex_to_vertex_rot);
		//massMatrix = AssembleMassElement(v);
		for( int i = 0 ; i < 3 ; ++i ) {
			for (int j = 0 ; j < 3 ; ++j ) {
				if (i < j) continue; // since stifness matrix is symmetric
				//wij = massMatrix(i, j) + delta_t * stiffnessMatrix(i, j);
				wij = delta_t * stiffnessMatrix(i, j);
				(*A)(v[i]->ID, v[j]->ID) += wij;
				if (i != j) {
					wji = delta_t * stiffnessMatrix(j, i);
					(*A)(v[j]->ID, v[i]->ID) += wji;
				}
			}
		}
	}
	return A;
}


std::shared_ptr<SparseMatrix<complex<double>>> AssembleDiagonalMassMatrix(Mesh *mesh) {
	std::shared_ptr<SparseMatrix<complex<double>>> A(new SparseMatrix<complex<double>>(mesh->numVertices(), mesh->numVertices()));
	Eigen::Matrix3d stiffnessMatrix, massMatrix;
	double wij;
	Vertex* v[3];
	for (Face& face : mesh->getFaces()) {
		v[0] = face.edge->vertex;
		v[1] = face.edge->next->vertex;
		v[2] = face.edge->next->next->vertex;
		double faceArea = 0.5 * ((v[1]->p - v[0]->p).cross(v[2]->p - v[0]->p)).norm();
		wij = faceArea / 3.0;
		(*A)(v[0]->ID, v[0]->ID) += wij;
		(*A)(v[1]->ID, v[1]->ID) += wij;
		(*A)(v[2]->ID, v[2]->ID) += wij;
	}
	return A;
}

void IplusMinvTimesA(std::shared_ptr<SparseMatrix<complex<double>>> M, std::shared_ptr<SparseMatrix<complex<double>>> A)
{
	auto numRows = A->numRows();
	for (int i = 0; i < numRows; ++i)
	{
		SparseMatrix<complex<double>>::RowIterator aIter = A->iterator(i);
		double oneOverVertexOneRingArea = 1.0 / (*M)(i, i).real();
		for (; !aIter.end(); ++aIter)
		{
			auto j = aIter.columnIndex();
			(*A)(i, j) *= oneOverVertexOneRingArea;
			if (i == j) {
				(*A)(i, j) += 1.0; // this completes the (I + M^-1 L)
			}
		}
	}
/*	
	auto numRows = A->numRows();
	for (int i = 0; i < numRows; ++i)
	{
		SparseMatrix<complex<double>>::RowIterator aIter = A->iterator(i);
		for (; !aIter.end(); ++aIter)
		{
			auto j = aIter.columnIndex();
			if (i == j) {
				(*A)(i, i) = (*M)(i, i) + (*A)(i, i);
			}
		}
	}
*/
}

bool is_constrained(std::set<int>& constraints, int vertex)
{
	return constraints.find(vertex) != constraints.end();
}

void PreFactor(std::shared_ptr<SparseMatrix<std::complex<double>>> A, std::set<int>& constraints, Eigen::SparseLU<Eigen::SparseMatrix<std::complex<double>>>& solver)
{

	Eigen::SparseMatrix<std::complex<double>> Lc = Eigen::SparseMatrix<std::complex<double>>(A->numRows(), A->numColumns());

	auto numRows = A->numRows();
	for (int i = 0; i < numRows; ++i)
	{
		if (!is_constrained(constraints, i))
		{
			SparseMatrix<std::complex<double>>::RowIterator aIter = A->iterator(i);
			for (; !aIter.end(); ++aIter)
			{
				auto j = aIter.columnIndex();
				Lc.insert(i, j) = (*A)(i, j);
			}
		}
		else
		{
			Lc.insert(i, i) = 1.0;
		}
	}

	Lc.makeCompressed();
	solver.compute(Lc);
	if (solver.info() != Eigen::Success) {
		std::cerr << "Error: " << "Prefactor failed." << std::endl;
		exit(1);
	}
}

int main(int argc, char* argv[])
{
	/**
	 * Load the FEM mesh
	 */
	//mesh.readFEM("lake_nodes.txt", "lake_elements.txt");
	mesh.readOBJ("cactus1.obj");
	mesh.CenterAndNormalize();
	mesh.computeNormals();

	// GLUT Window Initialization:
	glutInit (&argc, argv);
	glutInitWindowSize(g_viewportWidth, g_viewportHeight);
	glutInitDisplayMode( GLUT_RGB | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
	glutCreateWindow(WINDOW_TITLE);

	// Register callbacks:
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutMouseFunc(MouseButton);
	glutMotionFunc(MouseMotion);
	glutKeyboardUpFunc(KeyboardUpFunc);
	glutSpecialFunc(SpecialFunc);
	glutSpecialUpFunc(SpecialUpFunc);
	glutIdleFunc(Idle);
	atexit(DestroyWindow);

	InitializeDrawing();

	vertexBuffer.resize(mesh.numVertices());
	indexBuffer.resize(mesh.numFaces() * 3);

	/**
	 * Initialize the vertex-buffer for OpenGL rendering purposes
	 */
	for( Vertex& vertex : mesh.getVertices())
	{
		vertexBuffer.positions[vertex.ID] = vertex.p;
		vertexBuffer.normals[vertex.ID] = vertex.n;
		vertexBuffer.colors[vertex.ID] = valueToColor(0);
	}

	/**
	 * Initialize the index-buffer for OpenGL rendering purposes
	 */
	for (Face& face : mesh.getFaces()) {
		int i = face.ID;
		int	v1 = face.edge->vertex->ID;
		int	v2 = face.edge->next->vertex->ID;
		int	v3 = face.edge->next->next->vertex->ID;
		indexBuffer.faces[i * 3 + 0] = v1;
		indexBuffer.faces[i * 3 + 1] = v2;
		indexBuffer.faces[i * 3 + 2] = v3;
	}

	corner_angles.resize(mesh.numVertices());
	vertex_to_edge_angles.resize(mesh.numVertices());
	vertex_to_vertex_rot.resize(mesh.numVertices());

	double avgEdgeLength = computeEdgeLengths(&mesh);
	computeCornerAngles(&mesh, corner_angles);
	computeVertexToEdgeAngles(&mesh, corner_angles, vertex_to_edge_angles);
	computeVertexToVertexRotations(&mesh, vertex_to_edge_angles, vertex_to_vertex_rot);

	right_hand_side = Eigen::VectorXcd(mesh.numVertices());
	right_hand_side.setZero(); // solve laplace's equation where RHS is zero

	for(int i = 0; i < mesh.numVertices(); ++i){
		Vertex& vi = mesh.vertexAt(i);
		std::complex<double> X_i = std::complex<double>(0,0);
		right_hand_side[vi.ID] = X_i;
	}

	for( Vertex& vertex : mesh.getVertices())
	{
		if (vertex.p.norm() < 2.5e-2) {
			std::complex<double> U_i = std::complex<double>(cos(2*M_PI/3), sin(2*M_PI/3));
			right_hand_side(vertex.ID) = U_i;
			allconstraints.insert(vertex.ID);
			break;
		}
		if (vertex.p.z() > 0.83) {
			std::complex<double> U_i = std::complex<double>(cos(2*M_PI/3), sin(2*M_PI/3));
			right_hand_side(vertex.ID) = U_i;
			allconstraints.insert(vertex.ID);
			break;
		}

	}

	A = AssembleMatrix(&mesh, vertex_to_vertex_rot, avgEdgeLength*avgEdgeLength);
	std::shared_ptr<SparseMatrix<complex<double>>> M;
	M = AssembleDiagonalMassMatrix(&mesh);
	IplusMinvTimesA(M, A);

	PreFactor(A, allconstraints, solver);

	solutionU = solver.solve(right_hand_side);

	glutMainLoop();

	return 0;
}

void display()
{
	/*
	 *	matrices
	 */
	glViewport( 0, 0, g_viewportWidth, g_viewportHeight );
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	pickLoadMatrix();
	GLpick::g_frustumFar = 1000.0;
	GLpick::g_frustumNear = .1;
	gluPerspective( 60.0, (double)g_viewportWidth/(double)g_viewportHeight, GLpick::g_frustumNear, GLpick::g_frustumFar );
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glShadeModel(GL_SMOOTH);	//gouraud shading
	glClearDepth(1.0f);
	glClearColor( .75f, .75f, .75f, .0f );
	glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST );

	/*
	 *	estados
	 */
	glEnable(GL_CULL_FACE);		//face culling
	glCullFace( GL_BACK );
	glFrontFace( GL_CCW );
	glEnable(GL_DEPTH_TEST);	//z-buffer
	glDepthFunc(GL_LEQUAL);

	/*
	 *	iluminacion
	 */
	float		ambient[] = { .3f, .3f, .3f, 1.f };
	float		diffuse[] = { .3f, .3f, .3f, 1.f };
	float		position[] = { .0f, 0.f, 15.f, 1.f };
	float		specular[] = { 1.f, 1.f, 1.f };

	glLightfv( GL_LIGHT0, GL_AMBIENT, ambient );
	glLightfv( GL_LIGHT0, GL_DIFFUSE, diffuse );
	glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0);
	glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.0125);
	glEnable(  GL_LIGHT0   );
	glEnable(  GL_LIGHTING );
	//glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR, specular );
	glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, 50.f );

	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	glLoadIdentity();

	g_camera.glLookAt();

	glLightfv( GL_LIGHT0, /*GL_SPOT_DIRECTION*/GL_POSITION, position );

	glPushMatrix();

	rotorGLMult(g_modelRotor);

	if (GLpick::g_pickActive) glLoadName((GLuint)-1);

	double alpha = 1.0;

	//glEnable (GL_BLEND);
	//glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	//alpha = 0.5;

	//Mesh-Faces Rendering
	glPolygonMode( GL_FRONT_AND_BACK, GL_FILL /*GL_LINE GL_FILL GL_POINT*/);
	glEnable (GL_POLYGON_OFFSET_FILL);
	glPolygonOffset (1., 1.);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable( GL_COLOR_MATERIAL );
	if (GLpick::g_pickActive) glLoadName((GLuint)10);

	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glVertexPointer(3, GL_DOUBLE, 0, &vertexBuffer.positions[0]);
	glNormalPointer(GL_DOUBLE, 0, &vertexBuffer.normals[0]);
	glColorPointer(3, GL_DOUBLE, 0, &vertexBuffer.colors[0]);

	// draw the model
	glDrawElements(GL_TRIANGLES, indexBuffer.get_size(), GL_UNSIGNED_INT, &indexBuffer.faces[0]);
	// deactivate vertex arrays after drawing
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);

	if (g_showWires)
	{
		if (!GLpick::g_pickActive)
		{
			//Mesh-Edges Rendering (superimposed to faces)
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE /*GL_LINE GL_FILL GL_POINT*/);
			glColor4d(.5, .5, .5, alpha);
			glDisable(GL_LIGHTING);
			glEnableClientState(GL_VERTEX_ARRAY);
			glVertexPointer(3, GL_DOUBLE, 0, &vertexBuffer.positions[0]);
			// draw the model
			glDrawElements(GL_TRIANGLES, indexBuffer.get_size(), GL_UNSIGNED_INT, &indexBuffer.faces[0]);
			// deactivate vertex arrays after drawing
			glDisableClientState(GL_VERTEX_ARRAY);
			glEnable(GL_LIGHTING);
			glPolygonMode( GL_FRONT_AND_BACK, GL_FILL /*GL_LINE GL_FILL GL_POINT*/);

		}
	}

	glDisable( GL_COLOR_MATERIAL );
	glDisable(GL_POLYGON_OFFSET_FILL);

	//glDisable (GL_BLEND);
	float		green[] = { .0f, .5f, .0f, 1.f };
	float		red[] = { .5f, .0f, .0f, 1.f };
	float		blue[] = { .0f, .0f, .5f, 1.f };

	for(int i = 0; i < mesh.numVertices(); ++i){
		Vertex& vi = mesh.vertexAt(i);
		std::complex<double> R_i = solutionU[vi.ID] / abs(solutionU[vi.ID]);
		Eigen::Vector3d X_i = project(vi.edge->pair->vertex->p - vi.p, vi.n).normalized();
		Eigen::Quaterniond Q(Eigen::AngleAxisd(log(R_i).imag(), vi.n));
		Eigen::Vector3d U_i = 0.06 * Q._transformVector(X_i);
		DrawArrow(c3gaPoint(vi.p.x(), vi.p.y(), vi.p.z()), _vectorE3GA(U_i.x(), U_i.y(), U_i.z()));
	}

	glPopMatrix();

	glutSwapBuffers();
}

Eigen::Vector3d valueToColor( double d )
{
	static Eigen::Vector3d	c0 = Eigen::Vector3d( 1, 1, 1);
	static Eigen::Vector3d	c1 = Eigen::Vector3d( 1, 1, 0);
	static Eigen::Vector3d	c2 = Eigen::Vector3d( 0, 1, 0);
	static Eigen::Vector3d	c3 = Eigen::Vector3d( 0, 1, 1);
	static Eigen::Vector3d	c4 = Eigen::Vector3d( 0, 0, 1);

	if( d < 0.25 )
	{
		double alpha = (d - 0.0) / (0.25-0.0);
		return (1.0 - alpha) * c0 + alpha * c1;
	}
	else if( d < 0.5 )
	{
		double alpha = (d - 0.25) / (0.5-0.25);
		return (1.0 - alpha) * c1 + alpha * c2;
	}
	else if( d < 0.75 )
	{
		double alpha = (d - 0.5) / (0.75-0.5);
		return (1.0 - alpha) * c2 + alpha * c3;
	}
	else
	{
		double alpha = (d - 0.75) / (1.0-0.75);
		return (1.0 - alpha) * c3 + alpha * c4;
	}
}


void reshape(GLint width, GLint height)
{
	g_viewportWidth = width;
	g_viewportHeight = height;

	// redraw viewport
	glutPostRedisplay();
}

vectorE3GA mousePosToVector(int x, int y) {
	x -= g_viewportWidth / 2;
	y -= g_viewportHeight / 2;
	return _vectorE3GA((float)-x * e1 - (float)y * e2);
}

void MouseButton(int button, int state, int x, int y)
{
	g_rotateModel = false;

	if (button == GLUT_LEFT_BUTTON)
	{
		g_prevMousePos = mousePosToVector(x, y);

		GLpick::g_pickWinSize = 1;
		g_dragObject = pick(x, g_viewportHeight - y, display, &g_dragDistance);

		if(g_dragObject == -1 || g_dragObject == 10 )
		{
			vectorE3GA mousePos = mousePosToVector(x, y);
			g_rotateModel = true;

			if ((_Float(norm_e(mousePos)) / _Float(norm_e(g_viewportWidth * e1 + g_viewportHeight * e2))) < 0.2)
				g_rotateModelOutOfPlane = true;
			else g_rotateModelOutOfPlane = false;
		}
	}

	if (button == GLUT_RIGHT_BUTTON)
	{
		g_prevMousePos = mousePosToVector(x, y);

		GLpick::g_pickWinSize = 1;
		g_dragObject = pick(x, g_viewportHeight - y, display, &g_dragDistance);
	}
}

void MouseMotion(int x, int y)
{
	if (g_rotateModel )
	{
		// get mouse position, motion
		vectorE3GA mousePos = mousePosToVector(x, y);
		vectorE3GA motion = mousePos - g_prevMousePos;

		if (g_rotateModel)
		{
			// update rotor
			if (g_rotateModelOutOfPlane)
				g_modelRotor = exp(g_camera.rotateVel * (motion ^ e3) ) * g_modelRotor;
			else 
				g_modelRotor = exp(0.00001f * (motion ^ mousePos) ) * g_modelRotor;
		}

		// remember mouse pos for next motion:
		g_prevMousePos = mousePos;

		// redraw viewport
		glutPostRedisplay();
	}
}

void SpecialFunc(int key, int x, int y)
{
	switch(key) {
		case GLUT_KEY_F1 :
			{
				int mod = glutGetModifiers();
				if(mod == GLUT_ACTIVE_CTRL || mod == GLUT_ACTIVE_SHIFT )
				{
				}
			}
			break;
		case GLUT_KEY_UP:
			{
			}
			break;
		case GLUT_KEY_DOWN:
			{
			}
			break;
	}
}

void SpecialUpFunc(int key, int x, int y)
{
}

void KeyboardUpFunc(unsigned char key, int x, int y)
{
	if(key == 'w' || key == 'W')
	{
		g_showWires = !g_showWires;
		glutPostRedisplay();
	}
}

void Idle()
{
	// redraw viewport
}

void DestroyWindow()
{
	ReleaseDrawing();
}

