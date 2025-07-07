//1.将sqlSessionFactory声明为静态变量
//2.在静态代码块里使用sqlSessionFactoryBuilder创建sqlSessionFactory
//2.1使用Resources类加载mybatis核心配置文件
//2.2将读取的输入流放到sqlSessionFactoryBuilder里创建sqlSessionFactory


import fisher.mybatis.Dao.StudentDao;
import fisher.mybatis.pojo.Student;
import org.apache.ibatis.session.SqlSession;
import org.apache.ibatis.session.SqlSessionFactory;
import org.apache.ibatis.session.SqlSessionFactoryBuilder;
import org.apache.ibatis.io.Resources;
import org.junit.Test;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Test01 {

    private static SqlSessionFactory sqlSessionFactory;

    static {
        try {
            InputStream resourceAsStream = Resources.getResourceAsStream("mybatis-config.xml");
            sqlSessionFactory = new SqlSessionFactoryBuilder().build(resourceAsStream);
        } catch (IOException e) {
            throw new RuntimeException("加载mybatis配置文件失败", e);
        }
    }
    @Test//查询学生信息
    public void test01(){
        //1.创建会话
        SqlSession sqlSession = sqlSessionFactory.openSession();

        //2.调用
        Student student = sqlSession.selectOne("fisher.mybatis.Dao.StudentDao.selectStudentById", 100185L);
        //3.输出结果
        System.out.println(student);
        //4.关闭会话
        sqlSession.close();
    }

    //dao层
    @Test//查询所有学生信息
    public  void test02(){
        SqlSession sqlSession = sqlSessionFactory.openSession();

        StudentDao studentDao= sqlSession.getMapper(StudentDao.class);
        List<Student> students = studentDao.selectStudentList();
        for(Student student : students) {
            System.out.println(student);
        }
        sqlSession.close();

    }

    @Test//根据ID查询学生信息
    public void test03() {
        // 获取SqlSession对象，用于执行数据库操作
        SqlSession sqlSession = sqlSessionFactory.openSession();
        // 获取StudentDao接口的Mapper代理对象
        StudentDao studentDao = sqlSession.getMapper(StudentDao.class);
        // 调用DAO接口方法，查询ID为100004的学生信息
        Student student = studentDao.selectStudentById(100004);
        // 打印查询到的学生信息
        System.out.println(student);
        // 关闭SqlSession释放数据库连接资源
        sqlSession.close();
    }

    @Test//根据姓名查询学生信息
    public  void test04(){
        // 获取SqlSession对象，用于执行数据库操作
        SqlSession sqlSession = sqlSessionFactory.openSession();
        // 获取StudentDao接口的Mapper代理对象
        StudentDao studentDao = sqlSession.getMapper(StudentDao.class);
        // 调用DAO接口方法，根据姓名"小张张"查询学生信息
        // 返回结果是一个包含所有匹配学生的ArrayList集合
        ArrayList<Student> student = studentDao.selectStudentByName("小张张");
        // 遍历查询结果并打印每个学生的信息
        for (Student student1 : student)
            System.out.println(student1);
        // 关闭SqlSession释放数据库连接资源
        sqlSession.close();
    }

    @Test//插入学生信息
    public void test05() {
        // 获取SqlSession对象，用于执行数据库操作
        SqlSession sqlSession = sqlSessionFactory.openSession();
        // 获取StudentDao接口的Mapper代理对象
        StudentDao studentDao = sqlSession.getMapper(StudentDao.class);
        // 创建新的学生对象并初始化属性
        Student student = new Student(100005, "小王王", "2000-01-01", "北京市海淀区", 12345, 1, 20);
        // 调用DAO层插入方法，返回影响的行数
        int result = studentDao.insertStudent(student);
        // 根据返回结果输出操作状态
        if (result > 0)
            System.out.println("插入成功，影响行数：" + result);
        else
            System.out.println("插入失败");
        // 提交事务并关闭会话释放资源
        sqlSession.commit();
        sqlSession.close();
    }

    @Test//更新学生信息
    public  void test06(){
        // 获取SqlSession对象，用于执行数据库操作
        SqlSession sqlSession = sqlSessionFactory.openSession();
        // 获取StudentDao接口的Mapper代理对象
        StudentDao studentDao = sqlSession.getMapper(StudentDao.class);
        // 创建学生对象并初始化更新数据(ID为100188的学生将被更新)
        Student student = new Student(100188, "小王王", "2000-01-01", "北京市海淀区", 12345, 2, 20);
        // 执行更新操作，返回影响的行数
        int result = studentDao.updateStudent(student);
        // 根据更新结果输出提示信息
        if (result > 0)
            System.out.println("更新成功，影响行数：" + result);
        else
            System.out.println("更新失败");
        sqlSession.commit(); // 提交事务
        sqlSession.close();
    }

    @Test//删除学生信息--ID
    public void test07() {
        // 创建SqlSession对象，用于执行数据库操作
        SqlSession sqlSession = sqlSessionFactory.openSession();
        // 获取StudentDao接口的Mapper代理对象
        StudentDao studentDao = sqlSession.getMapper(StudentDao.class);
        // 执行删除操作，传入学生ID 100187，返回影响的行数
        int result = studentDao.deleteStudentById(100187);
        // 判断删除结果并输出相应提示信息
        if (result > 0)
            System.out.println("删除成功，影响行数：" + result);
        else
            System.out.println("删除失败");
        // 提交事务确保删除操作生效
        sqlSession.commit();
        // 关闭会话释放数据库连接资源
        sqlSession.close();
    }

    @Test//删除学生信息--姓名
    public void test08() {
        // 获取SqlSession对象，用于执行数据库操作
        SqlSession sqlSession = sqlSessionFactory.openSession();
        // 获取StudentDao接口的Mapper代理对象
        StudentDao studentDao = sqlSession.getMapper(StudentDao.class);
        // 执行删除操作，传入学生姓名"小王王"，返回影响的行数
        int result = studentDao.deleteStudentByName("小王王");
        // 判断删除结果并输出相应提示信息
        if (result > 0)
            System.out.println("删除成功，影响行数：" + result);
        else
            System.out.println("删除失败");
        // 提交事务确保删除操作生效
        sqlSession.commit(); // 提交事务
        // 关闭会话释放数据库连接资源
        sqlSession.close();
    }

    @Test//查询集合
    public void test09() {
        // 获取SqlSession对象，用于执行数据库操作
        SqlSession sqlSession = sqlSessionFactory.openSession();
        // 获取StudentDao接口的Mapper代理对象
        StudentDao studentDao = sqlSession.getMapper(StudentDao.class);
        // 创建查询条件对象
        Student student = new Student();
        // 设置学生姓名查询条件(模糊查询)
        student.setStudentName("张");
        // 执行条件查询，返回符合条件的学生列表
        List<Student> students = studentDao.getStudentByCondition(student);
        // 遍历查询结果并打印每个学生信息
        for (Student s : students) {
            System.out.println(s);
        }
        // 关闭会话释放数据库连接资源
        sqlSession.close();
    }

    @Test//Map入参
    public void test10() {
        // 获取SqlSession对象，用于执行数据库操作
        SqlSession sqlSession = sqlSessionFactory.openSession();
        // 获取StudentDao接口的Mapper代理对象
        StudentDao studentDao = sqlSession.getMapper(StudentDao.class);
        // 创建查询条件Map
        Map<String, Object> map = Map.of("studentName", "王", "gradeId", 1);
        // 执行Map条件查询，返回符合条件的学生列表
        List<Student> students = studentDao.getStudentByMap(map);
        // 遍历查询结果并打印每个学生信息
        for (Student s : students) {
            System.out.println(s);
        }
        // 关闭会话释放数据库连接资源
        sqlSession.close();
    }

    @Test//返回Map
    public void test11() {
        // 获取SqlSession对象，用于执行数据库操作
        SqlSession sqlSession = sqlSessionFactory.openSession();
        // 获取StudentDao接口的Mapper代理对象
        StudentDao studentDao = sqlSession.getMapper(StudentDao.class);
        // 执行查询操作，获取ID为100004的学生信息，并返回Map格式结果
        Map<String, Object> studentMapRes = studentDao.getStudentMapRes(100004);
        // 打印查询结果Map
        System.out.println(studentMapRes);
        // 关闭会话释放数据库连接资源
        sqlSession.close();
    }

    @Test//返回多个map
    public void test12() {
        // 获取SqlSession对象，用于执行数据库操作
        SqlSession sqlSession = sqlSessionFactory.openSession();
        // 获取StudentDao接口的Mapper代理对象
        StudentDao studentDao = sqlSession.getMapper(StudentDao.class);
        Student student = new Student();
        student.setStudentName("王"); // 设置查询条件，模糊查询学生姓名包含"王"
        // 执行查询操作，获取所有学生信息，并返回Map列表
        List<Map<String, Object>> studentMapList = studentDao.getStudentMapList(student);
        // 遍历查询结果并打印每个学生信息的Map
        studentMapList.forEach(System.out::println);
        // 关闭会话释放数据库连接资源
        sqlSession.close();
    }
}


