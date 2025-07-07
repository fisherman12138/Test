package fisher.mybatis.Dao;

import fisher.mybatis.pojo.Student;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public interface StudentDao {
    // 定义方法来查询学生信息
    public ArrayList<Student> selectStudentList();// 查询所有学生信息
    public Student selectStudentById(long studentId);// 根据ID查询学生信息
    public ArrayList<Student> selectStudentByName(String studentName);// 根据姓名查询学生信息
    public int insertStudent(Student student);// 插入学生信息
    public int updateStudent(Student student);// 更新学生信息
    public int deleteStudentById(long studentId);// 根据ID删除学生信息
    public int deleteStudentByName(String studentName);// 根据姓名删除学生信息
    public List<Student> getStudentByCondition(Student student);// 根据条件查询学生信息
    public List<Student> getStudentByMap(Map<String, Object> map);// 根据Map条件查询学生信息
    public Map<String, Object> getStudentMapRes(long studentId);// 根据ID查询学生信息并返回Map
    public List<Map<String,Object>> getStudentMapList(Student student);// 查询所有学生信息并返回Map列表
}
